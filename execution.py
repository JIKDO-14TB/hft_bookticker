# hft_backtest_engine/execution.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class Order:
    order_id: int
    symbol: str
    side: int                 # +1 buy, -1 sell
    size: float               # notional (cash 단위)
    order_type: str           # "market" | "limit"
    price: Optional[float]
    created_ts: pd.Timestamp
    expire_ts: Optional[pd.Timestamp] = None
    status: str = "PENDING"


@dataclass
class FillResult:
    filled: bool
    fill_price: Optional[float]
    fee_bp: float = 0.0
    fee_amt: float = 0.0


class ExecutionEngine:
    """
    ExecutionEngine
    ---------------
    - latency_ms: 주문 활성화 지연
    - slippage_bp: 시장가(or taker fill) 체결가 슬리피지
    - fee_bp_market: 시장가 수수료 (bp)
    - fee_bp_limit: 지정가 수수료 (bp)
    """

    def __init__(
        self,
        latency_ms: int = 5,
        slippage_bp: float = 1.0,
        fee_bp_market: float = 5.0,
        fee_bp_limit: float = 2.0,
    ):
        self.latency_ms = int(latency_ms)
        self.slippage_bp = float(slippage_bp)
        self.fee_bp_market = float(fee_bp_market)
        self.fee_bp_limit = float(fee_bp_limit)

    def order_active_ts(self, order: Order) -> pd.Timestamp:
        return order.created_ts + pd.Timedelta(milliseconds=self.latency_ms)

    def try_fill(self, order: Order, trade) -> FillResult:
        """
        trade: aggTrades itertuples row (ts, price, ...)
        """
        trade_price = float(trade.price)

        # =========================
        # 1) Fill 결정 + fill_price
        # =========================
        if order.order_type == "market":
            # ✅ 시장가: 슬리피지 적용
            fill_price = self._apply_slippage(trade_price, order.side)
            fee_bp = self.fee_bp_market

        elif order.order_type == "limit":
            # ✅ 지정가: 조건 불충족이면 미체결
            if order.price is None:
                return FillResult(False, None)

            limit_px = float(order.price)

            # buy limit: 시장가가 limit 이하로 내려와야 체결
            if order.side == 1 and trade_price > limit_px:
                return FillResult(False, None)

            # sell limit: 시장가가 limit 이상으로 올라와야 체결
            if order.side == -1 and trade_price < limit_px:
                return FillResult(False, None)

            # ✅ 단순화: 체결가는 trade_price로
            # (원하면: buy는 min(trade_price, limit_px), sell은 max(...)로 바꿔도 됨)
            fill_price = trade_price

            # ✅ 지정가: 보통 slippage 0으로 둠 (원하면 적용 가능)
            fee_bp = self.fee_bp_limit

        else:
            raise ValueError(f"Unknown order_type: {order.order_type}")

        # =========================
        # 2) Fee 계산 (cash 단위)
        # =========================
        fee_amt = float(order.size) * (fee_bp * 1e-4)

        return FillResult(
            filled=True,
            fill_price=float(fill_price),
            fee_bp=float(fee_bp),
            fee_amt=float(fee_amt),
        )

    def _apply_slippage(self, price: float, side: int) -> float:
        """
        side=+1 (buy): 더 비싸게 체결
        side=-1 (sell): 더 싸게 체결
        """
        slip = float(price) * self.slippage_bp * 1e-4
        return float(price) + float(side) * slip
