# hft_backtest_engine/feature_store.py
from __future__ import annotations

from typing import Dict, Optional, Deque
from collections import deque

import pandas as pd

from hft_backtest_engine.features import (
    VPINCalculator,
    OFICalculator,
    compute_trade_count_spike,
    compute_qr,
)


class FeatureStore:
    """
    FeatureStore (STATEFUL + FAST + VPIN-GATED)
    ------------------------------------------
    - VPIN은 tick마다 증분 계산
    - VPIN gate가 열릴 때만 OFI / QR / TC 계산
    - VPIN bucket TTL 지나면 나머지 지표 상태 전부 폐기
    """

    def __init__(
        self,
        symbol: str,
        signal_interval_seconds: int = 5 * 60,
        vpin_bucket_volume: float = 1e6,
        vpin_history: int = 100,
        vpin_gate_threshold: float = 0.7,
        tc_window: int = 60,
        ofi_window_minutes: int = 5,
        ofi_z_window: int = 30,
        keep_book_minutes: int = 10,
    ):
        self.symbol = symbol
        self.signal_interval_seconds = signal_interval_seconds
        self.vpin_gate_threshold = float(vpin_gate_threshold)

        # =========================
        # Stateful calculators
        # =========================
        self.vpin_calc = VPINCalculator(
            bucket_volume=vpin_bucket_volume,
            history=vpin_history,
        )
        self.ofi_calc = OFICalculator(
            window_minutes=ofi_window_minutes,
            z_window=ofi_z_window,
        )

        self.tc_window = tc_window
        self.ofi_window_minutes = ofi_window_minutes
        self.keep_book_minutes = max(keep_book_minutes, ofi_window_minutes + 1)

        # =========================
        # FAST buffers
        # =========================
        self._kline_rows: Deque[dict] = deque(maxlen=self.tc_window + 2)

        self._book_snaps: Deque[pd.DataFrame] = deque(maxlen=50_000)
        self._last_book_ts: Optional[pd.Timestamp] = None
        self._last_book_snap: Optional[pd.DataFrame] = None

        # cache
        self.last_feature_ts: Optional[pd.Timestamp] = None
        self.cached_features: Optional[Dict[str, float]] = None

    # =====================================================
    # Update API
    # =====================================================

    def update_trade(self, trade_row) -> None:
        """aggTrades tick 1개"""
        self.vpin_calc.update_trade(trade_row)

    def update_kline(self, kline_df: pd.DataFrame) -> None:
        if kline_df is None or kline_df.empty:
            return

        row = kline_df.iloc[-1]
        self._kline_rows.append({
            "open_ts": row["open_ts"],
            "trades": float(row["trades"]),
        })

    def update_book(self, book_df: pd.DataFrame) -> None:
        if book_df is None or book_df.empty:
            return

        snap_ts = book_df["ts"].iloc[0]
        if self._last_book_ts is not None and snap_ts == self._last_book_ts:
            return

        self._book_snaps.append(book_df)
        self._last_book_ts = snap_ts
        self._last_book_snap = book_df

        cutoff = snap_ts - pd.Timedelta(minutes=self.keep_book_minutes)
        while self._book_snaps and self._book_snaps[0]["ts"].iloc[0] < cutoff:
            self._book_snaps.popleft()

    # =====================================================
    # Feature computation (VPIN-GATED)
    # =====================================================

    def should_compute(self, ts: pd.Timestamp) -> bool:
        if self.last_feature_ts is None:
            return True
        return (ts - self.last_feature_ts).total_seconds() >= self.signal_interval_seconds

    def compute_features(self, ts: pd.Timestamp) -> Dict[str, float]:
        # -------------------------
        # 1️⃣ VPIN (always)
        # -------------------------
        vpin = self.vpin_calc.get_value()
        vpin_cdf = float(vpin.get("vpin_cdf", 0.0))

        # -------------------------
        # 2️⃣ VPIN gate CLOSED → reset & early return
        # -------------------------
        if vpin_cdf < self.vpin_gate_threshold:
            # 🔥 나머지 지표 상태 전부 폐기
            self._book_snaps.clear()
            self._last_book_snap = None
            self._last_book_ts = None
            self._kline_rows.clear()

            # OFI 상태 리셋 (가장 중요)
            self.ofi_calc = OFICalculator(
                window_minutes=self.ofi_window_minutes,
                z_window=self.ofi_calc.z_window,
            )

            features = {
                "ts": ts,
                "vpin_raw": float(vpin.get("vpin_raw", float("nan"))),
                "vpin_cdf": vpin_cdf,
                "tc": 0.0,
                "z_tc": 0.0,
                "n_cdf": 0.5,
                "ofi_raw": 0.0,
                "z_ofi": 0.0,
                "qr": 0.0,
            }

            self.last_feature_ts = ts
            self.cached_features = features
            return features

        # -------------------------
        # 3️⃣ Trade Count Spike
        # -------------------------
        if len(self._kline_rows) < self.tc_window + 1:
            tc = {"tc": 0.0, "z_tc": 0.0, "n_cdf": 0.5}
        else:
            kline_df = pd.DataFrame(self._kline_rows).sort_values("open_ts")
            tc = compute_trade_count_spike(kline_df, window=self.tc_window)

        # -------------------------
        # 4️⃣ OFI (recent window only)
        # -------------------------
        if not self._book_snaps:
            ofi = {"ofi_raw": 0.0, "z_ofi": 0.0}
        else:
            book_df = pd.concat(self._book_snaps, ignore_index=True).sort_values("ts")
            start = ts - pd.Timedelta(minutes=self.ofi_window_minutes)
            book_slice = book_df[(book_df["ts"] >= start) & (book_df["ts"] <= ts)]

            if book_slice["ts"].nunique() < 2:
                ofi = {"ofi_raw": 0.0, "z_ofi": 0.0}
            else:
                ofi = self.ofi_calc.update(book_slice)

        # -------------------------
        # 5️⃣ QR (latest snapshot)
        # -------------------------
        qr = compute_qr(self._last_book_snap) if self._last_book_snap is not None else 0.0

        features = {
            "ts": ts,
            "vpin_raw": float(vpin.get("vpin_raw", float("nan"))),
            "vpin_cdf": vpin_cdf,
            "tc": float(tc.get("tc", 0.0)),
            "z_tc": float(tc.get("z_tc", 0.0)),
            "n_cdf": float(tc.get("n_cdf", 0.5)),
            "ofi_raw": float(ofi.get("ofi_raw", 0.0)),
            "z_ofi": float(ofi.get("z_ofi", 0.0)),
            "qr": float(qr),
        }

        self.last_feature_ts = ts
        self.cached_features = features
        return features

    def get_features(self, ts: pd.Timestamp) -> Dict[str, float]:
        if self.cached_features is None or self.should_compute(ts):
            return self.compute_features(ts)
        return self.cached_features

