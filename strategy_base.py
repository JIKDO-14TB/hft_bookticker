from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from hft_backtest_engine.feature_store import FeatureStore


# =====================================================
# Position
# =====================================================

@dataclass
class Position:
    symbol: str
    size: float
    side: int            # +1 long, -1 short
    entry_price: float
    entry_ts: pd.Timestamp


# =====================================================
# Strategy State
# =====================================================

@dataclass
class StrategyState:
    cash: float
    positions: Dict[str, Position]
    current_ts: pd.Timestamp


# =====================================================
# Order
# =====================================================

@dataclass
class Order:
    symbol: str
    side: int
    size: float
    order_type: str      # "market" | "limit"
    price: Optional[float]


# =====================================================
# Config
# =====================================================

@dataclass
class StrategyConfig:
    base_target_bp: float = 25.0
    partial_take_bp: float = 15.0
    max_holding_seconds: int = 4 * 60

    max_leverage_notional: float = 1.0
    z_trade_count_threshold: float = 2.5

    use_ofi: bool = False
    use_qr: bool = True

    # 🔥 NEW
    use_stop_loss: bool = True


# =====================================================
# Strategy
# =====================================================

class Strategy:
    def __init__(
        self,
        symbol: str,
        feature_store: FeatureStore,
        config: StrategyConfig,
        initial_capital: float = 100.0,
    ):
        self.symbol = symbol
        self.feature_store = feature_store
        self.config = config
        self.initial_capital = initial_capital

        # 🔥 NEW: exit counters
        self.exit_counts = {
            "STOP": 0,
            "TIME": 0,
            "TP": 0,
        }

    # =================================================
    # Score (direction only)
    # =================================================

    def compute_score(self, tick, state: StrategyState) -> float:
        feats = self.feature_store.get_features(tick.ts)

        z_tc = feats.get("z_tc", 0.0)
        if abs(z_tc) < self.config.z_trade_count_threshold:
            return 0.0

        if self.config.use_qr:
            qr = feats.get("qr", 0.0)
            return float(np.sign(qr))

        if self.config.use_ofi:
            z_ofi = feats.get("z_ofi", 0.0)
            return float(np.sign(z_ofi))

        return 0.0

    # =================================================
    # Tick-level exit
    # =================================================

    def on_tick(self, tick, state: StrategyState) -> List[Order]:
        orders: List[Order] = []

        pos = state.positions.get(self.symbol)
        if pos is None:
            return orders

        ts = tick.ts
        holding = (ts - pos.entry_ts).total_seconds()

        mid = 0.5 * (
            float(tick.best_bid_price) +
            float(tick.best_ask_price)
        )

        # =====================
        # 1️⃣ STOP LOSS
        # =====================
        if self.config.use_stop_loss:
            stop_bp = self.config.partial_take_bp * 1e-4

            if pos.side == 1 and mid <= pos.entry_price * (1.0 - stop_bp):
                self.exit_counts["STOP"] += 1
                orders.append(
                    Order(self.symbol, -pos.side, pos.size, "market", None)
                )
                return orders

            if pos.side == -1 and mid >= pos.entry_price * (1.0 + stop_bp):
                self.exit_counts["STOP"] += 1
                orders.append(
                    Order(self.symbol, -pos.side, pos.size, "market", None)
                )
                return orders

        # =====================
        # 2️⃣ TIME STOP
        # =====================
        if holding >= self.config.max_holding_seconds:
            self.exit_counts["TIME"] += 1
            orders.append(
                Order(self.symbol, -pos.side, pos.size, "market", None)
            )

        return orders

    # =================================================
    # Entry
    # =================================================

    def on_signal(self, tick, state: StrategyState) -> List[Order]:
        orders: List[Order] = []

        if self.symbol in state.positions:
            return orders

        score = self.compute_score(tick, state)
        if score == 0.0:
            return orders

        side = 1 if score > 0 else -1
        size = state.cash * self.config.max_leverage_notional

        bid = float(tick.best_bid_price)
        ask = float(tick.best_ask_price)
        price = 0.5 * (bid + ask)

        # entry
        orders.append(Order(self.symbol, side, size, "market", None))

        # TP1
        tp1_price = price * (1.0 + side * self.config.base_target_bp * 1e-4)
        orders.append(
            Order(self.symbol, -side, size * 0.5, "limit", tp1_price)
        )

        # TP2
        tp2_price = price * (1.0 + side * self.config.partial_take_bp * 1e-4)
        orders.append(
            Order(self.symbol, -side, size * 0.5, "limit", tp2_price)
        )

        return orders

    # =================================================
    # Engine hook
    # =================================================

    def on_trade(self, tick, state: StrategyState) -> List[Order]:
        out = self.on_tick(tick, state)
        if out:
            return out
        return self.on_signal(tick, state)






