from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd
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
    partial_closed: bool = False


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
# Order (엔진과 인터페이스)
# =====================================================

@dataclass
class Order:
    symbol: str
    side: int
    size: float
    order_type: str              # "market" | "limit"
    price: float | None


# =====================================================
# Config
# =====================================================

@dataclass
class StrategyConfig:
    # score weights
    w_ofi: float = 0.7
    w_qr: float = 0.3

    # gates
    use_vpin_gate: bool = False
    vpin_gate_threshold: float = 0.7

    # exits
    base_target_bp: float = 30.0        # 1차 take-profit limit bp (절반)
    partial_take_bp: float = 30.0       # 남은 물량 market 청산 트리거 bp
    max_holding_seconds: int = 4 * 60

    # scaling
    max_leverage_notional: float = 1.0
    score_window: int = 288

    # 안전장치: score가 너무 작으면 “의미없는 목표가”가 되어 바로 청산되는 상황 방지용
    min_score_for_exit: float = 0.1


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
        self.signal_interval_seconds = signal_interval_seconds

        self.feature_logs = []

    # -------------------------------------------------
    # Helpers (✅ 누락되어 있던 함수들 추가)
    # -------------------------------------------------
    def _should_recompute_signal(self, ts: pd.Timestamp, state: StrategyState) -> bool:
        if state.last_signal_ts is None:
            return True
        return (ts - state.last_signal_ts).total_seconds() >= self.signal_interval_seconds

    def _score_to_weight(self, score: float, state: StrategyState) -> float:
        """
        최근 score_history 기반 min/max 정규화로 [-1, 1] weight 산출
        """
        state.score_history.append(float(score))
        if len(state.score_history) > self.config.score_window:
            state.score_history = state.score_history[-self.config.score_window:]

        mn = min(state.score_history)
        mx = max(state.score_history)
        if mx == mn:
            return 0.0

        norm01 = (score - mn) / (mx - mn)
        weight = 2.0 * norm01 - 1.0
        return max(-1.0, min(1.0, float(weight)))

    # -------------------------------------------------
    # Score
    # -------------------------------------------------
    def compute_score(self, trade, state: StrategyState) -> float:
        ts = trade.ts
        feats = self.feature_store.get_features(ts)

        vpin_cdf = feats.get("vpin_cdf", float("nan"))
        z_ofi = feats.get("z_ofi", float("nan"))
        qr = feats.get("qr", float("nan"))
        n_cdf = feats.get("n_cdf", float("nan"))

        if any(pd.isna(x) for x in [vpin_cdf, z_ofi, qr, n_cdf]):
            score = 0.0
        else:
            if self.config.use_vpin_gate and vpin_cdf < self.config.vpin_gate_threshold:
                score = 0.0
            else:
                raw = self.config.w_ofi * z_ofi + self.config.w_qr * qr
                score = raw * n_cdf

        self.feature_logs.append({"ts": ts, "score": float(score)})
        return float(score)

    # -------------------------------------------------
    # Tick-level exit
    # - 남은 물량은 market 청산
    # - (주의) 절반 지정가 체결 여부는 엔진이 포지션 size를 직접 줄이진 않으므로
    #   “부분청산”을 전략에서 직접 반영하려면 fills 기반 업데이트가 필요함.
    #   지금은 단순화 버전: partial 트리거 또는 시간초과 시 '현재 포지션 size' 전량 청산
    # -------------------------------------------------
    def on_tick(self, trade, state: StrategyState) -> List[Order]:
        orders: List[Order] = []
        pos = state.positions.get(self.symbol)
        if pos is None:
            return orders

        price = float(trade.price)
        ts = trade.ts

        pnl_bp = pos.side * (price - pos.entry_price) / pos.entry_price * 1e4
        holding = (ts - pos.entry_ts).total_seconds()

        # score 너무 작으면 “실질 목표가”가 너무 낮아져서 즉시 청산될 수 있음 → 방지
        score_abs = max(abs(state.last_score), self.config.min_score_for_exit)

        # 남은 물량 market 청산 트리거
        if pnl_bp >= self.config.partial_take_bp * score_abs:
            orders.append(Order(self.symbol, -pos.side, pos.size, "market", None))
        elif holding >= self.config.max_holding_seconds:
            orders.append(Order(self.symbol, -pos.side, pos.size, "market", None))

        return orders

    # -------------------------------------------------
    # Signal-level entry
    # - 전량 시장가 진입
    # - 동시에 절반 목표가 지정가 제출
    # -------------------------------------------------
    def on_signal(self, trade, state: StrategyState) -> List[Order]:
        orders: List[Order] = []
        ts = trade.ts
        price = float(trade.price)

        if self.symbol in state.positions:
            return orders

        if not self._should_recompute_signal(ts, state):
            return orders

        score = self.compute_score(trade, state)
        state.last_signal_ts = ts
        state.last_score = score

        weight = self._score_to_weight(score, state)
        if weight == 0.0:
            return orders

        size = state.cash * self.config.max_leverage_notional * abs(weight)
        side = 1 if weight > 0 else -1

        # 1) 시장가 진입
        orders.append(Order(self.symbol, side, size, "market", None))

        # 2) 절반 목표가 지정가 (미리 걸기)
        score_abs = max(abs(score), self.config.min_score_for_exit)
        target_bp = self.config.base_target_bp * score_abs
        target_price = price * (1 + side * target_bp * 1e-4)

        orders.append(Order(self.symbol, -side, size * 0.5, "limit", float(target_price)))

        return orders

    # -------------------------------------------------
    # Wrapper (BacktestEngine 호환)
    # -------------------------------------------------
    def on_trade(self, trade, state: StrategyState) -> List[Order]:
        out = self.on_tick(trade, state)
        if out:
            return out
        return self.on_signal(trade, state)




