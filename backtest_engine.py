# hft_backtest_engine/backtest_engine.py
from __future__ import annotations

from typing import Dict, Optional
import itertools
import pandas as pd

from hft_backtest_engine.data_loader import DataLoader
from hft_backtest_engine.strategy_base import (
    Strategy,
    StrategyState,
    Position,
    Order as StrategyOrder,
)
from hft_backtest_engine.execution import (
    ExecutionEngine,
    Order as ExecOrder,
    FillResult,
)
from hft_backtest_engine.feature_store import FeatureStore

COMPUTE_DELAY_MS = 10


class BacktestEngine:
    """
    변경점 요약
    ----------
    ✅ limit 주문 5초 만료(expire_ts) 개념 제거
      - TP limit은 GTC처럼 계속 살아있게 둠
      - 강제 market 전환 로직 삭제

    ✅ 포지션이 "완전 청산"되는 순간,
       해당 심볼의 남아있는 limit 주문들을 정리(optional but recommended)
    """

    def __init__(
        self,
        loader: DataLoader,
        strategy: Strategy,
        execution: ExecutionEngine,
        feature_store: FeatureStore,
        initial_capital: float = 100.0,
        verbose: bool = True,
    ):
        self.loader = loader
        self.strategy = strategy
        self.execution = execution
        self.feature_store = feature_store
        self.verbose = verbose

        self.state = StrategyState(
            cash=initial_capital,
            positions={},
            current_ts=None,
        )

        # active_orders에는 ExecOrder만 저장
        self.active_orders: Dict[int, ExecOrder] = {}
        self.order_id_gen = itertools.count(1)

        self.fills = []
        self.closed_trades = []

        # book snapshot 중복 push 방지
        self._last_pushed_book_ts: Optional[pd.Timestamp] = None
        self._last_seen_book_ts: Optional[pd.Timestamp] = None

    def run_day(self, symbol: str, ymd: str):
        # 0) load
        df = self.loader.load_aggtrades_day(symbol, ymd)
        if df is None or df.empty:
            if self.verbose:
                print(f"[WARN] no aggTrades: {symbol} {ymd}")
            return

        try:
            book_df = self.loader.load_bookdepth_day(symbol, ymd)
        except Exception as e:
            if self.verbose:
                print(f"[WARN] no bookDepth: {symbol} {ymd} ({e})")
            book_df = pd.DataFrame()

        try:
            klines = self.loader.load_klines_1m_day(symbol, ymd)
        except Exception as e:
            if self.verbose:
                print(f"[WARN] no klines_1m: {symbol} {ymd} ({e})")
            klines = pd.DataFrame()

        trades = df.sort_values("ts").reset_index(drop=True)

        # 1) book snapshot index (ts -> row indices)
        book_ts_list = []
        book_groups = None
        if not book_df.empty:
            book_df = book_df.sort_values(["ts", "percentage"]).reset_index(drop=True)
            book_groups = book_df.groupby("ts", sort=True).indices  # dict: ts -> np.array idx
            book_ts_list = sorted(book_groups.keys())
        book_ptr = 0

        # 2) kline index (open_ts -> row) : 룩어헤드 방지(직전 분만 push)
        kline_map = None
        if not klines.empty:
            klines = klines.sort_values("open_ts").reset_index(drop=True)
            kline_map = klines.set_index("open_ts", drop=False)

        last_min = None

        for trade in trades.itertuples(index=False):
            ts = trade.ts
            self.state.current_ts = ts

            # (A) FeatureStore tick update
            self.feature_store.update_trade(trade)

            # (B) kline push (룩어헤드 방지: 새 minute 진입 시 직전 minute만 push)
            if kline_map is not None:
                cur_min = ts.floor("1min")
                if last_min is None:
                    last_min = cur_min
                elif cur_min > last_min:
                    last_completed_min = last_min
                    last_min = cur_min

                    if last_completed_min in kline_map.index:
                        self.feature_store.update_kline(kline_map.loc[[last_completed_min]])

            # (C) book snapshot push (ts까지 도달한 snapshot들 중 마지막 1개만 push)
            if book_groups is not None and book_ts_list:
                while book_ptr < len(book_ts_list) and book_ts_list[book_ptr] <= ts:
                    self._last_seen_book_ts = book_ts_list[book_ptr]
                    book_ptr += 1

                last_seen = self._last_seen_book_ts
                if last_seen is not None and self._last_pushed_book_ts != last_seen:
                    idxs = book_groups[last_seen]
                    snapshot = book_df.loc[idxs]
                    self.feature_store.update_book(snapshot)
                    self._last_pushed_book_ts = last_seen

            # (D) fill existing orders
            self._process_active_orders(trade)

            # (E-1) tick exit only
            exit_orders = self.strategy.on_tick(trade=trade, state=self.state)
            for o in exit_orders:
                self._submit_order(o, ts)

            # (E-2) signal entry only
            entry_orders = self.strategy.on_signal(trade=trade, state=self.state)
            for o in entry_orders:
                self._submit_order(o, ts)

    # =====================================================
    # 주문 제출: StrategyOrder -> ExecOrder
    # =====================================================
    def _submit_order(self, proto_order: StrategyOrder, ts: pd.Timestamp):
        oid = next(self.order_id_gen)
        created_ts = ts + pd.Timedelta(milliseconds=COMPUTE_DELAY_MS)

        # ✅ 변경: limit expire_ts 제거 (TP limit은 GTC처럼 계속 유지)
        expire_ts = None

        order = ExecOrder(
            order_id=oid,
            symbol=proto_order.symbol,
            side=proto_order.side,
            size=float(proto_order.size),
            order_type=proto_order.order_type,
            price=None if proto_order.price is None else float(proto_order.price),
            created_ts=created_ts,
            expire_ts=expire_ts,
            status="PENDING",
        )

        self.active_orders[oid] = order

        if self.verbose:
            print(
                f"[ORDER] {created_ts} {order.symbol} "
                f"{order.order_type.upper()} id={oid} (compute+{COMPUTE_DELAY_MS}ms)"
            )

    # =====================================================
    # active_orders 처리
    # =====================================================
    def _process_active_orders(self, trade):
        to_remove = []

        for oid, order in list(self.active_orders.items()):
            if trade.ts < self.execution.order_active_ts(order):
                continue

            # ✅ 변경: limit 만료/강제 market 로직 삭제
            # - TP limit은 그냥 살아있다가, 가격 도달 시 체결
            # - 포지션 종료 시점(시장가 청산)에는 _apply_fill에서 잔여 주문 정리

            fill: FillResult = self.execution.try_fill(order, trade)
            if fill.filled:
                self._apply_fill(order, fill, trade.ts)
                to_remove.append(oid)

        for oid in to_remove:
            self.active_orders.pop(oid, None)

    # =====================================================
    # 체결 반영 (fee 포함)
    # =====================================================
    def _apply_fill(self, order: ExecOrder, fill: FillResult, ts: pd.Timestamp):
        symbol = order.symbol
        price = float(fill.fill_price)

        fee_bp = float(getattr(fill, "fee_bp", 0.0))
        fee_amt = float(getattr(fill, "fee_amt", 0.0))

        # =========================
        # 진입 (포지션 없으면 ENTER)
        # =========================
        if symbol not in self.state.positions:
            pos = Position(
                symbol=symbol,
                size=order.size,
                side=order.side,
                entry_price=price,
                entry_ts=ts,
            )
            # entry fee 저장 (exit 때 total fee에 포함)
            setattr(pos, "entry_fee_amt", fee_amt)

            self.state.positions[symbol] = pos
            self.state.cash -= (order.size + fee_amt)

            self.fills.append({
                "ts": ts,
                "symbol": symbol,
                "fill_type": "ENTER",
                "side": order.side,
                "price": price,
                "size": order.size,
                "order_type": order.order_type,
                "order_id": order.order_id,
                "fee_bp": fee_bp,
                "fee_amt": fee_amt,
                "gross_pnl": 0.0,
                "net_pnl": -fee_amt,
            })

            if self.verbose:
                print(f"[FILL ENTER] {ts} {symbol} price={price:.4f} fee_bp={fee_bp:.2f}")

            return

        # =========================
        # EXIT (포지션 있으면 청산)
        # - 여기서는 "부분청산"도 가능하게 처리
        #   order.size 만큼만 줄이고,
        #   size가 0에 가까워지면 포지션 제거 + 남은 주문 정리
        # =========================
        pos = self.state.positions[symbol]
        entry_fee_amt = float(getattr(pos, "entry_fee_amt", 0.0))

        # 부분청산 가능: exit_size = order.size (단, pos.size보다 크지 않게 방어)
        exit_size = float(min(order.size, pos.size))
        if exit_size <= 0:
            return

        gross_pnl = pos.side * (price - pos.entry_price) / pos.entry_price * exit_size

        # fee 배분:
        # - exit fee는 이번 exit_size에 대한 fee_amt (ExecutionEngine이 order.size 기반으로 계산하므로 OK)
        # - entry fee는 "마지막 완전청산 때" 한 번에 빼면 총합은 맞지만,
        #   부분청산에서 net을 보고 싶으면 entry_fee를 비례배분하는 게 더 깔끔함
        # 여기서는 비례배분(권장): entry_fee_alloc = entry_fee * (exit_size / original_size)
        # original_size를 따로 저장하지 않으면 현 pos.size + exit_size로 복원 가능
        original_size_before = pos.size
        total_size_before = original_size_before  # 현재 남아있던 크기
        # 이 체결은 total_size_before 중 exit_size 만큼 청산
        # entry_fee_amt는 "전체 포지션"에 대한 비용이므로, 비례배분
        entry_fee_alloc = entry_fee_amt * (exit_size / max(total_size_before, 1e-12))

        total_fee_amt = entry_fee_alloc + fee_amt
        net_pnl = gross_pnl - total_fee_amt

        # cash update:
        # entry 때 이미 (size + entry_fee) 빠졌고,
        # exit 때는 (exit_size + gross_pnl - exit_fee) 더해주면 net 반영됨.
        self.state.cash += (exit_size + gross_pnl - fee_amt)

        self.fills.append({
            "ts": ts,
            "symbol": symbol,
            "fill_type": "EXIT",
            "side": -pos.side,
            "price": price,
            "size": exit_size,
            "order_type": order.order_type,
            "order_id": order.order_id,
            "fee_bp": fee_bp,
            "fee_amt": fee_amt,
            "entry_fee_alloc": entry_fee_alloc,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
        })

        # 포지션 크기 감소 + entry fee도 남은 만큼 유지
        pos.size -= exit_size
        remaining_ratio = pos.size / max(total_size_before, 1e-12)
        setattr(pos, "entry_fee_amt", entry_fee_amt * remaining_ratio)

        # 완전 청산이면 closed_trades 기록 + 남은 주문 정리
        if pos.size <= 1e-12:
            self.state.positions.pop(symbol, None)

            self.closed_trades.append({
                "symbol": symbol,
                "entry_ts": pos.entry_ts,
                "exit_ts": ts,
                "side": pos.side,
                "entry_price": pos.entry_price,
                "exit_price": price,
                "size": exit_size,  # 마지막 청산 size
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "holding_sec": (ts - pos.entry_ts).total_seconds(),
                "win": net_pnl > 0,
            })

            # ✅ (중요) 포지션 종료 시 남아있는 TP limit 같은 주문들 제거
            for oid2, o2 in list(self.active_orders.items()):
                if o2.symbol == symbol:
                    self.active_orders.pop(oid2, None)

            if self.verbose:
                print(f"[FILL EXIT ] {ts} {symbol} gross={gross_pnl:.4f} net={net_pnl:.4f} fee_bp={fee_bp:.2f}")
        else:
            if self.verbose:
                print(f"[FILL PARTIAL] {ts} {symbol} size={exit_size:.4f} rem={pos.size:.4f} gross={gross_pnl:.4f} net={net_pnl:.4f}")



