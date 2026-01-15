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
    size: float                  # 남은 notional
    side: int                    # +1 long, -1 short
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

    last_signal_ts: Optional[pd.Timestamp] = None
    last_score: float = 0.0
    score_history: List[float] = field(default_factory=list)


# =====================================================
# Order (BacktestEngine 인터페이스)
# =====================================================

@dataclass
class Order:
    symbol: str
    side: int
    size: float
    order_type: str              # "market" | "limit"
    price: Optional[float]


# =====================================================
# Config
# =====================================================

@dataclass
class StrategyConfig:
    # score weights
    w_ofi: float = 0.7
    w_qr: float = 0.3

    # gates
    use_vpin_gate: bool = True
    vpin_gate_threshold: float = 0.7

    # exits (bp 기준, score 무관)
    base_target_bp: float = 20.0       # TP1
    partial_take_bp: float = 40.0      # TP2
    max_holding_seconds: int = 4 * 60

    # sizing
    max_leverage_notional: float = 1.0
    score_window: int = 288


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
        signal_interval_seconds: int = 5 * 60,
    ):
        self.symbol = symbol
        self.feature_store = feature_store
        self.config = config
        self.initial_capital = initial_capital
        self.signal_interval_seconds = int(signal_interval_seconds)

        self.feature_logs = []

    # =================================================
    # Helpers
    # =================================================

    def _should_recompute_signal(self, ts: pd.Timestamp, state: StrategyState) -> bool:
        if state.last_signal_ts is None:
            return True
        return (ts - state.last_signal_ts).total_seconds() >= self.signal_interval_seconds

    def _score_to_weight(self, score: float, state: StrategyState) -> float:
        """
        최근 score_history 기반 min-max 정규화 → [-1, 1]
        ✅ 첫 신호(표본 1개)에서 mx==mn으로 weight=0이 되어 진입이 막히는 문제 해결:
        - score != 0 이면 sign(score)로 ±1을 반환해서 "첫 진입"을 허용
        """
        state.score_history.append(float(score))

        # ✅ 첫 1개(또는 2개 미만)일 때는 sign 기반으로 진입 허용
        if len(state.score_history) < 2:
            return float(np.sign(score)) if score != 0 else 0.0

        # window 유지
        if len(state.score_history) > self.config.score_window:
            state.score_history = state.score_history[-self.config.score_window:]

        mn = min(state.score_history)
        mx = max(state.score_history)
        if mx == mn:
            # 이 케이스는 거의 없음(2개 이상인데 모두 같은 값)
            return 0.0

        norm01 = (score - mn) / (mx - mn)
        weight = 2.0 * norm01 - 1.0
        return float(max(-1.0, min(1.0, weight)))

    # =================================================
    # Score
    # =================================================

    def compute_score(self, tick, state: StrategyState) -> float:
        ts = tick.ts
        feats = self.feature_store.get_features(ts)

        if feats.get("vpin_ready", 0.0) < 1.0:
            return 0.0

        z_ofi = feats.get("z_ofi", 0.0)
        qr = feats.get("qr", 0.0)
        n_cdf = feats.get("n_cdf", 0.5)

        score = (self.config.w_ofi * z_ofi + self.config.w_qr * qr) * n_cdf
        return float(score)

    # =================================================
    # Tick-level exit (bookTicker)
    # =================================================

    def on_tick(self, tick, state: StrategyState) -> List[Order]:
        orders: List[Order] = []

        pos = state.positions.get(self.symbol)
        if pos is None:
            return orders

        ts = tick.ts
        holding = (ts - pos.entry_ts).total_seconds()

        if holding >= self.config.max_holding_seconds:
            orders.append(Order(self.symbol, -pos.side, pos.size, "market", None))

        return orders

    # =================================================
    # Signal-level entry
    # =================================================

    def on_signal(self, tick, state: StrategyState) -> List[Order]:
        orders: List[Order] = []

        if self.symbol in state.positions:
            return orders

        ts = tick.ts
        if not self._should_recompute_signal(ts, state):
            return orders

        score = self.compute_score(tick, state)
        state.last_signal_ts = ts
        state.last_score = score

        if score == 0.0:
            return orders

        side = 1 if score > 0 else -1
        size = state.cash * self.config.max_leverage_notional

        # mid price
        bid = float(tick.best_bid_price)
        ask = float(tick.best_ask_price)
        price = 0.5 * (bid + ask)

        # 1) 시장가 진입
        orders.append(Order(self.symbol, side, size, "market", None))

        # 2) TP1
        tp1_price = price * (1.0 + side * self.config.base_target_bp * 1e-4)
        orders.append(Order(self.symbol, -side, size * 0.5, "limit", tp1_price))

        # 3) TP2
        tp2_price = price * (1.0 + side * self.config.partial_take_bp * 1e-4)
        orders.append(Order(self.symbol, -side, size * 0.5, "limit", tp2_price))

        return orders

    # =================================================
    # BacktestEngine wrapper
    # =================================================

    def on_trade(self, tick, state: StrategyState) -> List[Order]:
        out = self.on_tick(tick, state)
        if out:
            return out
        return self.on_signal(tick, state)






