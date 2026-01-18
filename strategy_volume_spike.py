# strategy_volume_spike.py (새 파일)
from hft_backtest_engine.strategy_base import Strategy, StrategyConfig

class VolumeSpikeStrategy(Strategy):
    def __init__(self,
                 symbol: str,
                 feature_store,
                 config: StrategyConfig,
                 initial_capital: float,
                 z_trade_count_threshold: float = 2.0,
                 z_volume_threshold: float = 2.0):
        super().__init__(symbol, feature_store, config, initial_capital)
        self.z_tc_thresh = z_trade_count_threshold
        self.z_vol_thresh = z_volume_threshold

    def _should_recompute_signal(self, ts) -> bool:
        """신호를 언제 계산할지 결정한다. z‑score 임계값을 초과한 경우에만 True"""
        # FeatureStore가 z_tc와 z_vol을 최신으로 업데이트했는지 확인한다.
        feats = self.feature_store.get_features(ts)
        if feats is None:
            return False
        # z‑score 임계값 체크
        return (feats.get('z_tc', 0) >= self.z_tc_thresh) and \
               (feats.get('z_vol', 0) >= self.z_vol_thresh)

    def compute_score(self, feats: dict) -> float:
        """매수/매도 점수 계산. q_buy와 q_sell을 이용해 방향을 결정."""
        # 매수수량과 매도수량 차이를 이용하여 방향 결정
        q_buy = feats.get('q_buy', 0)
        q_sell = feats.get('q_sell', 0)
        # buy pressure > sell pressure → 양수, 그 반대는 음수
        return q_buy - q_sell

    def on_signal(self, ts, tick) -> None:
        """신호 발생시 호출. _should_recompute_signal이 True일 때만 진입."""
        if not self._should_recompute_signal(ts):
            return
        feats = self.feature_store.get_features(ts)
        score = self.compute_score(feats)
        # 점수가 0이면 신호 없음
        if score == 0:
            return
        # 포지션 진입 – 자본의 100%를 롱/숏
        weight = 1.0 if score > 0 else -1.0
        # 기존 Strategy의 방법을 이용해 체결가 산출 및 포지션 구축
        self._place_new_order(ts, tick, weight)
