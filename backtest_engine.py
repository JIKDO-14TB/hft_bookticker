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
    BookTicker 기반 BacktestEngine
    - 엔진 클럭: bookTicker(ts)
    - 체결 판단: bookTicker(best bid/ask)
    - VPIN: aggTrades 보조 업데이트 (tick.ts까지 누적)

    성능 개선 포인트
    --------------
    ✅ aggTrades를 list로 만들지 않고 iterator 스트리밍으로 처리
    """

    def __init__(
        self,
        loader: DataLoader,
        strategy: Strategy,
        execution: ExecutionEngine,
        feature_store: FeatureStore,
        initial_capital: float = 100.0,
        verbose: bool = True,
        debug_ticks: int = 0,        # 처음 N틱 디버그 출력
        max_ticks: Optional[int] = None,  # 하루 중 일부만 테스트(빠른 디버깅용)
    ):
        self.loader = loader
        self.strategy = strategy
        self.execution = execution
        self.feature_store = feature_store
        self.verbose = bool(verbose)
        self.debug_ticks = int(debug_ticks)
        self.max_ticks = None if max_ticks is None else int(max_ticks)

        self.state = StrategyState(cash=float(initial_capital), positions={}, current_ts=None)

        self.active_orders: Dict[int, ExecOrder] = {}
        self.order_id_gen = itertools.count(1)

        self.fills = []
        self.closed_trades = []

    # =====================================================
    # Main
    # =====================================================
    def run_day(self, symbol: str, ymd: str):
        # 0) load
        ticks = self.loader.load_bookticker_day(symbol, ymd)
        if ticks is None or ticks.empty:
            if self.verbose:
                print(f"[WARN] empty bookTicker: {symbol} {ymd}")
            return

        try:
            trades_df = self.loader.load_aggtrades_day(symbol, ymd)
        except Exception as e:
            if self.verbose:
                print(f"[WARN] no aggTrades: {symbol} {ymd} ({e})")
            trades_df = pd.DataFrame()

        try:
            klines = self.loader.load_klines_1m_day(symbol, ymd)
        except Exception as e:
            if self.verbose:
                print(f"[WARN] no klines_1m: {symbol} {ymd} ({e})")
            klines = pd.DataFrame()

        # 정렬 (안전)
        ticks = ticks.sort_values("ts").reset_index(drop=True)

        # ✅ aggTrades 스트리밍 준비
        trade_iter = None
        cur_trade = None
        if trades_df is not None and not trades_df.empty:
            trades_df = trades_df.sort_values("ts").reset_index(drop=True)
            trade_iter = trades_df.itertuples(index=False)
            cur_trade = next(trade_iter, None)

        # 1m kline map (lookahead 방지: 직전 분만 push)
        kline_map = None
        if klines is not None and not klines.empty:
            klines = klines.sort_values("open_ts").reset_index(drop=True)
            kline_map = klines.set_index("open_ts", drop=False)
        last_min = None

        # main loop
        n_ticks = len(ticks)
        if self.max_ticks is not None:
            n_ticks = min(n_ticks, self.max_ticks)

        for i, tick in enumerate(ticks.itertuples(index=False)):
            if self.max_ticks is not None and i >= self.max_ticks:
                break

            ts = tick.ts
            self.state.current_ts = ts

            # (1) aggTrades -> VPIN update (tick.ts까지 누적)
            while cur_trade is not None and cur_trade.ts <= ts:
                self.feature_store.update_trade(cur_trade)
                cur_trade = next(trade_iter, None) if trade_iter is not None else None

            # (2) bookTicker -> OFI/QR update
            self.feature_store.update_bookticker(tick)

            # (3) kline push: 새 minute 진입 시 직전 minute만 push
            if kline_map is not None:
                cur_min = ts.floor("1min")
                if last_min is None:
                    last_min = cur_min
                elif cur_min > last_min:
                    last_completed_min = last_min
                    last_min = cur_min
                    if last_completed_min in kline_map.index:
                        self.feature_store.update_kline(kline_map.loc[[last_completed_min]])

            # ✅ 디버그: 처음 N틱만 출력
            if self.debug_ticks > 0 and i < self.debug_ticks:
                feats = self.feature_store.get_features(ts)
                pos_on = "Y" if symbol in self.state.positions else "N"
                print(
                    f"[TICK {i:>6}/{n_ticks}] ts={ts} "
                    f"bid={float(tick.best_bid_price):.4f} ask={float(tick.best_ask_price):.4f} "
                    f"vpin_cdf={feats.get('vpin_cdf', 0.0):.3f} z_ofi={feats.get('z_ofi', 0.0):.3f} "
                    f"qr={feats.get('qr', 0.0):.3f} n_cdf={feats.get('n_cdf', 0.0):.3f} "
                    f"pos={pos_on} cash={self.state.cash:.4f}"
                )

            # (4) fill orders
            self._process_active_orders(tick)

            # (5) strategy (tick 기반)
            exit_orders = self.strategy.on_tick(tick=tick, state=self.state)
            for o in exit_orders:
                self._submit_order(o, ts)

            entry_orders = self.strategy.on_signal(tick=tick, state=self.state)
            for o in entry_orders:
                self._submit_order(o, ts)

        if self.verbose:
            print(f"[DONE] {symbol} {ymd} ticks={n_ticks} fills={len(self.fills)} closed={len(self.closed_trades)}")

    # =====================================================
    # Submit order: StrategyOrder -> ExecOrder
    # =====================================================
    def _submit_order(self, proto_order: StrategyOrder, ts: pd.Timestamp):
        # size 방어: 0이면 주문 자체를 만들지 않는다
        if proto_order.size is None or float(proto_order.size) <= 0:
            if self.verbose:
                print(f"[SKIP ORDER] size<=0 symbol={proto_order.symbol} type={proto_order.order_type}")
            return

        oid = next(self.order_id_gen)
        created_ts = ts + pd.Timedelta(milliseconds=COMPUTE_DELAY_MS)

        order = ExecOrder(
            order_id=oid,
            symbol=proto_order.symbol,
            side=int(proto_order.side),
            size=float(proto_order.size),
            order_type=str(proto_order.order_type),
            price=None if proto_order.price is None else float(proto_order.price),
            created_ts=created_ts,
            expire_ts=None,
            status="PENDING",
        )
        self.active_orders[oid] = order

        if self.verbose:
            print(
                f"[ORDER] ts={created_ts} oid={oid} sym={order.symbol} "
                f"type={order.order_type} side={order.side} size={order.size:.4f} px={order.price}"
            )

    # =====================================================
    # Active orders fill
    # =====================================================
    def _process_active_orders(self, tick):
        to_remove = []
        for oid, order in list(self.active_orders.items()):
            if tick.ts < self.execution.order_active_ts(order):
                continue

            fill: FillResult = self.execution.try_fill(order, tick)
            if fill.filled:
                self._apply_fill(order, fill, tick.ts)
                to_remove.append(oid)

        for oid in to_remove:
            self.active_orders.pop(oid, None)

    # =====================================================
    # Apply fill (fee 포함)
    # =====================================================
    def _apply_fill(self, order: ExecOrder, fill: FillResult, ts: pd.Timestamp):
        symbol = order.symbol
        price = float(fill.fill_price)
        fee_bp = float(getattr(fill, "fee_bp", 0.0))
        fee_amt = float(getattr(fill, "fee_amt", 0.0))
        reason = str(getattr(fill, "reason", ""))

        # ENTER
        if symbol not in self.state.positions:
            pos = Position(symbol=symbol, size=order.size, side=order.side, entry_price=price, entry_ts=ts)
            setattr(pos, "entry_fee_amt", fee_amt)

            self.state.positions[symbol] = pos
            self.state.cash -= (order.size + fee_amt)

            self.fills.append({
                "ts": ts, "symbol": symbol, "fill_type": "ENTER",
                "side": order.side, "price": price, "size": order.size,
                "order_type": order.order_type, "order_id": order.order_id,
                "fee_bp": fee_bp, "fee_amt": fee_amt,
                "gross_pnl": 0.0, "net_pnl": -fee_amt,
                "reason": reason,
            })

            if self.verbose:
                print(f"[FILL ENTER] ts={ts} sym={symbol} px={price:.4f} fee={fee_amt:.6f} reason={reason}")
            return

        # EXIT / PARTIAL
        pos = self.state.positions[symbol]
        entry_fee_amt = float(getattr(pos, "entry_fee_amt", 0.0))

        exit_size = float(min(order.size, pos.size))
        if exit_size <= 0:
            return

        gross_pnl = pos.side * (price - pos.entry_price) / pos.entry_price * exit_size

        total_size_before = pos.size
        entry_fee_alloc = entry_fee_amt * (exit_size / max(total_size_before, 1e-12))
        net_pnl = gross_pnl - (entry_fee_alloc + fee_amt)

        self.state.cash += (exit_size + gross_pnl - fee_amt)

        self.fills.append({
            "ts": ts, "symbol": symbol, "fill_type": "EXIT",
            "side": -pos.side, "price": price, "size": exit_size,
            "order_type": order.order_type, "order_id": order.order_id,
            "fee_bp": fee_bp, "fee_amt": fee_amt,
            "entry_fee_alloc": entry_fee_alloc,
            "gross_pnl": gross_pnl, "net_pnl": net_pnl,
            "reason": reason,
        })

        pos.size -= exit_size
        remaining_ratio = pos.size / max(total_size_before, 1e-12)
        setattr(pos, "entry_fee_amt", entry_fee_amt * remaining_ratio)

        # 포지션 완전 종료
        if pos.size <= 1e-12:
            self.state.positions.pop(symbol, None)

            self.closed_trades.append({
                "symbol": symbol,
                "entry_ts": pos.entry_ts,
                "exit_ts": ts,
                "side": pos.side,
                "entry_price": pos.entry_price,
                "exit_price": price,
                "size": exit_size,
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "holding_sec": (ts - pos.entry_ts).total_seconds(),
                "win": net_pnl > 0,
            })

            # 포지션 종료 시 잔여 주문 제거
            for oid2, o2 in list(self.active_orders.items()):
                if o2.symbol == symbol:
                    self.active_orders.pop(oid2, None)

            if self.verbose:
                print(f"[FILL EXIT ] ts={ts} sym={symbol} gross={gross_pnl:.6f} net={net_pnl:.6f} reason={reason}")
        else:
            if self.verbose:
                print(
                    f"[FILL PART] ts={ts} sym={symbol} exit={exit_size:.4f} rem={pos.size:.4f} "
                    f"gross={gross_pnl:.6f} net={net_pnl:.6f} reason={reason}"
                )
