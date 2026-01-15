from __future__ import annotations

from typing import Dict, Optional, Deque
from collections import deque
import pandas as pd

from hft_backtest_engine.features import (
    VPINCalculator,
    OFIBestLevelCalculator,
    compute_trade_count_spike,
    compute_qr_from_bookticker,
)


class FeatureStore:
    def __init__(
        self,
        symbol: str,
        signal_interval_seconds: int = 5 * 60,
        vpin_bucket_volume: float = 1e6,
        vpin_history: int = 288,
        vpin_gate_threshold: float = 0.7,
        vpin_min_buckets: int = 20,   # ✅ 핵심
        tc_window: int = 60,
        ofi_window_minutes: int = 5,
        ofi_z_window: int = 30,
    ):
        self.symbol = symbol
        self.signal_interval_seconds = int(signal_interval_seconds)
        self.vpin_gate_threshold = float(vpin_gate_threshold)
        self.vpin_min_buckets = int(vpin_min_buckets)

        self.vpin_calc = VPINCalculator(
            bucket_volume=vpin_bucket_volume,
            history=vpin_history,
        )

        self.ofi_window_minutes = int(ofi_window_minutes)
        self.ofi_z_window = int(ofi_z_window)
        self.ofi_calc = OFIBestLevelCalculator(
            window_minutes=self.ofi_window_minutes,
            z_window=self.ofi_z_window,
        )

        self.tc_window = int(tc_window)

        self._kline_rows: Deque[dict] = deque(maxlen=self.tc_window + 2)
        self._last_booktick = None

        self.last_feature_ts: Optional[pd.Timestamp] = None
        self.cached_features: Optional[Dict[str, float]] = None

    # -----------------
    # Updates
    # -----------------
    def update_trade(self, trade_row) -> None:
        self.vpin_calc.update_trade(trade_row)

    def update_bookticker(self, tick) -> None:
        self._last_booktick = tick

        v = self.vpin_calc.get_value(now_ts=tick.ts)
        gate_open = (
            v["n_buckets"] >= self.vpin_min_buckets
            and v["vpin_cdf"] >= self.vpin_gate_threshold
        )

        if gate_open:
            self.ofi_calc.update_bookticker(tick)

    def update_kline(self, kline_df: pd.DataFrame) -> None:
        if kline_df is None or kline_df.empty:
            return
        row = kline_df.iloc[-1]
        self._kline_rows.append(
            {"open_ts": row["open_ts"], "trades": float(row["trades"])}
        )

    # -----------------
    # Features
    # -----------------
    def compute_features(self, ts: pd.Timestamp) -> Dict[str, float]:
        vpin = self.vpin_calc.get_value(now_ts=ts)
        n_buckets = int(vpin["n_buckets"])

        gate_open = (
            n_buckets >= self.vpin_min_buckets
            and vpin["vpin_cdf"] >= self.vpin_gate_threshold
        )

        if not gate_open:
            self.ofi_calc = OFIBestLevelCalculator(
                window_minutes=self.ofi_window_minutes,
                z_window=self.ofi_z_window,
            )
            self._kline_rows.clear()

        if len(self._kline_rows) < self.tc_window + 1:
            tc = {"tc": 0.0, "z_tc": 0.0, "n_cdf": 0.5}
        else:
            tc = compute_trade_count_spike(
                pd.DataFrame(self._kline_rows).sort_values("open_ts"),
                window=self.tc_window,
            )

        feats = {
            "ts": ts,
            "vpin_raw": float(vpin["vpin_raw"]),
            "vpin_cdf": float(vpin["vpin_cdf"]),
            "n_vpin_buckets": n_buckets,
            "vpin_ready": float(gate_open),
            "tc": float(tc["tc"]),
            "z_tc": float(tc["z_tc"]),
            "n_cdf": float(tc["n_cdf"]),
            "ofi_raw": float(self.ofi_calc.last_ofi_raw),
            "z_ofi": float(self.ofi_calc.last_z_ofi),
            "qr": float(
                compute_qr_from_bookticker(self._last_booktick)
                if self._last_booktick is not None
                else 0.0
            ),
        }

        self.last_feature_ts = ts
        self.cached_features = feats
        return feats

    def get_features(self, ts: pd.Timestamp) -> Dict[str, float]:
        if self.cached_features is None:
            return self.compute_features(ts)
        if (ts - self.last_feature_ts).total_seconds() >= self.signal_interval_seconds:
            return self.compute_features(ts)
        return self.cached_features
