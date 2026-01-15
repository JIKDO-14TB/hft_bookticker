from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal
import pandas as pd


# =====================================================
# Data classes
# =====================================================
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
    reason: str = ""          # 디버그용


# =====================================================
# Execution Engine (Hybrid-clock aware)
# =====================================================
class ExecutionEngine:
    """
    Hybrid-clock 대응 ExecutionEngine

    MARKET
      - BUY  -> ask_last (+ slippage)
      - SELL -> bid_last (- slippage)

    LIMIT (touch-fill, 구간 min/max 기준)
      - BUY  -> ask_min <= limit_price
      - SELL -> bid_max >= limit_price
    """

    def __init__(
        self,
        latency_ms: int = 5,
        slippage_bp: float = 1.0,
        fee_bp_market: float = 5.0,
        fee_bp_limit: float = 2.0,
        fill_price_mode: Literal["limit", "touch", "improve"] = "limit",
        debug: bool = False,
    ):
        self.latency_ms = int(latency_ms)
        self.slippage_bp = float(slippage_bp)
        self.fee_bp_market = float(fee_bp_market)
        self.fee_bp_limit = float(fee_bp_limit)
        self.fill_price_mode = fill_price_mode
        self.debug = bool(debug)

    # -------------------------------------------------
    def order_active_ts(self, order: Order) -> pd.Timestamp:
        return order.created_ts + pd.Timedelta(milliseconds=self.latency_ms)

    # -------------------------------------------------
    def _get_prices(self, tick):
        """
        Hybrid-safe price extractor
        """
        # last (항상 필요)
        bid_last = float(getattr(tick, "best_bid_price"))
        ask_last = float(getattr(tick, "best_ask_price"))

        # hybrid clock용 (있으면 사용)
        bid_max = float(getattr(tick, "best_bid_price_max", bid_last))
        ask_min = float(getattr(tick, "best_ask_price_min", ask_last))

        return bid_last, ask_last, bid_max, ask_min

    # -------------------------------------------------
    def try_fill(self, order: Order, tick) -> FillResult:
        bid_last, ask_last, bid_max, ask_min = self._get_prices(tick)

        # -------------------------
        # 방어 코드
        # -------------------------
        if bid_last <= 0 or ask_last <= 0:
            return FillResult(False, None, reason="bad_tick_price<=0")

        if bid_last > ask_last:
            return FillResult(False, None, reason="bad_tick_bid>ask")

        if order.size is None or float(order.size) <= 0:
            return FillResult(False, None, reason="bad_order_size<=0")

        # =========================
        # MARKET
        # =========================
        if order.order_type == "market":
            raw_price = ask_last if order.side == 1 else bid_last
            fill_price = self._apply_slippage(raw_price, order.side)
            fee_bp = self.fee_bp_market
            fee_amt = float(order.size) * (fee_bp * 1e-4)

            if self.debug:
                print(
                    f"[EXEC][MARKET] ts={tick.ts} oid={order.order_id} "
                    f"side={order.side} fill={fill_price:.4f}"
                )

            return FillResult(True, fill_price, fee_bp, fee_amt, reason="market")

        # =========================
        # LIMIT (touch-fill, hybrid)
        # =========================
        if order.order_type == "limit":
            if order.price is None:
                return FillResult(False, None, reason="limit_missing_price")

            limit_px = float(order.price)

            # BUY limit: ask_min <= limit
            if order.side == 1:
                if ask_min > limit_px:
                    return FillResult(False, None, reason="buy_limit_not_touched")

                if self.fill_price_mode == "limit":
                    fill_price = limit_px
                elif self.fill_price_mode == "touch":
                    fill_price = min(limit_px, ask_last)
                else:  # improve
                    fill_price = min(limit_px, ask_min)

            # SELL limit: bid_max >= limit
            else:
                if bid_max < limit_px:
                    return FillResult(False, None, reason="sell_limit_not_touched")

                if self.fill_price_mode == "limit":
                    fill_price = limit_px
                elif self.fill_price_mode == "touch":
                    fill_price = max(limit_px, bid_last)
                else:  # improve
                    fill_price = max(limit_px, bid_max)

            fee_bp = self.fee_bp_limit
            fee_amt = float(order.size) * (fee_bp * 1e-4)

            if self.debug:
                print(
                    f"[EXEC][LIMIT] ts={tick.ts} oid={order.order_id} "
                    f"side={order.side} limit={limit_px:.4f} fill={fill_price:.4f}"
                )

            return FillResult(True, fill_price, fee_bp, fee_amt, reason="limit_touch")

        raise ValueError(f"Unknown order_type: {order.order_type}")

    # -------------------------------------------------
    def _apply_slippage(self, price: float, side: int) -> float:
        slip = float(price) * self.slippage_bp * 1e-4
        return float(price) + float(side) * slip



