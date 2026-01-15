from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm


# ======================================================
# A. VPIN (Stateful, bucketed, TTL decay)
# ======================================================

class VPINCalculator:
    def __init__(
        self,
        bucket_volume: float,
        history: int = 288,
        bucket_ttl_seconds: int = 30 * 60,   # ✅ 기본 30분
    ):
        self.bucket_volume = float(bucket_volume)
        self.history = int(history)
        self.bucket_ttl = pd.Timedelta(seconds=int(bucket_ttl_seconds))

        self.acc_vol = 0.0
        self.buy_vol = 0.0
        self.sell_vol = 0.0

        # (completed_ts, imbalance)
        self.bucket_imbalances: Deque[Tuple[pd.Timestamp, float]] = deque()

    def update_trade(self, trade) -> None:
        q = float(trade.quantity)
        is_buyer_maker = bool(trade.is_buyer_maker)
        ts = trade.ts

        self.acc_vol += q
        if is_buyer_maker:
            self.sell_vol += q
        else:
            self.buy_vol += q

        if self.acc_vol >= self.bucket_volume:
            denom = max(self.buy_vol + self.sell_vol, 1e-12)
            imb = abs(self.buy_vol - self.sell_vol) / denom

            self.bucket_imbalances.append((ts, float(imb)))

            self.acc_vol = 0.0
            self.buy_vol = 0.0
            self.sell_vol = 0.0

            while len(self.bucket_imbalances) > self.history:
                self.bucket_imbalances.popleft()

    def get_value(self, now_ts: Optional[pd.Timestamp] = None) -> dict:
        if now_ts is not None:
            cutoff = now_ts - self.bucket_ttl
            while self.bucket_imbalances and self.bucket_imbalances[0][0] < cutoff:
                self.bucket_imbalances.popleft()

        n = len(self.bucket_imbalances)
        if n == 0:
            return {
                "vpin_raw": np.nan,
                "vpin_cdf": 0.0,
                "n_buckets": 0,
            }

        vpin_raw = float(self.bucket_imbalances[-1][1])
        hist = np.asarray([v for _, v in self.bucket_imbalances], dtype="float64")
        vpin_cdf = float(np.mean(hist <= vpin_raw))

        return {
            "vpin_raw": vpin_raw,
            "vpin_cdf": vpin_cdf,
            "n_buckets": int(n),
        }


# ======================================================
# B. Trade Count Spike (1m) - Stateless
# ======================================================

def compute_trade_count_spike(klines_1m: pd.DataFrame, window: int = 60) -> dict:
    """
    최신 1분 trades가 과거 window분 대비 얼마나 이례적인지
    required: ["open_ts", "trades"]
    """
    if klines_1m is None or klines_1m.empty or "trades" not in klines_1m.columns:
        return {"tc": 0.0, "z_tc": 0.0, "n_cdf": 0.5}

    if len(klines_1m) < window + 1:
        tc_cur = float(pd.to_numeric(klines_1m["trades"].iloc[-1], errors="coerce") or 0.0)
        return {"tc": tc_cur, "z_tc": 0.0, "n_cdf": 0.5}

    tc_series = pd.to_numeric(klines_1m["trades"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    tc_cur = float(tc_series[-1])

    hist = tc_series[-window - 1:-1]
    mu = float(hist.mean())
    sigma = float(hist.std(ddof=0))

    z = 0.0 if sigma == 0 else (tc_cur - mu) / sigma
    n_cdf = float(norm.cdf(z))
    return {"tc": tc_cur, "z_tc": float(z), "n_cdf": n_cdf}


# ======================================================
# C. OFI (Best-Level, bookTicker 기반)
# ======================================================

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class _BestQuote:
    price: float
    qty: float


class OFIBestLevelCalculator:
    """
    bookTicker 최우선호가(best bid/ask) 기반 OFI (Cont et al.)

    - 엔진 클럭: bookTicker ts
    - 상태 유지: prev bid/ask, rolling impact window
    """

    def __init__(self, window_minutes: int = 5, z_window: int = 30):
        self.window_minutes = int(window_minutes)
        self.z_window = int(z_window)

        self.prev_bid: Optional[_BestQuote] = None
        self.prev_ask: Optional[_BestQuote] = None

        self.impact_window: Deque[Tuple[pd.Timestamp, float]] = deque()
        self.ofi_history: Deque[float] = deque(maxlen=self.z_window)

        # ✅ FeatureStore가 읽을 "최신 값 캐시"
        self.last_ofi_raw: float = 0.0
        self.last_z_ofi: float = 0.0
        self.last_e_total: float = 0.0

    def update_bookticker(self, tick) -> None:
        """
        tick: bookTicker row (itertuples or Series)
        required:
          - ts
          - best_bid_price, best_bid_qty
          - best_ask_price, best_ask_qty
        """

        # Series / itertuples 둘 다 지원
        if isinstance(tick, (dict, pd.Series)):
            ts = tick["ts"]
            bid_p = float(tick["best_bid_price"])
            bid_q = float(tick["best_bid_qty"])
            ask_p = float(tick["best_ask_price"])
            ask_q = float(tick["best_ask_qty"])
        else:
            ts = tick.ts
            bid_p = float(tick.best_bid_price)
            bid_q = float(tick.best_bid_qty)
            ask_p = float(tick.best_ask_price)
            ask_q = float(tick.best_ask_qty)

        # -------------------------
        # 최초 tick: 기준점 설정
        # -------------------------
        if self.prev_bid is None:
            self.prev_bid = _BestQuote(bid_p, bid_q)
            self.prev_ask = _BestQuote(ask_p, ask_q)

            self.last_ofi_raw = 0.0
            self.last_z_ofi = 0.0
            self.last_e_total = 0.0
            return

        # -------------------------
        # Bid impact
        # -------------------------
        pb_prev, qb_prev = self.prev_bid.price, self.prev_bid.qty
        if bid_p > pb_prev:
            e_b = +bid_q
        elif bid_p < pb_prev:
            e_b = -qb_prev
        else:
            e_b = bid_q - qb_prev

        # -------------------------
        # Ask impact
        # -------------------------
        pa_prev, qa_prev = self.prev_ask.price, self.prev_ask.qty
        if ask_p < pa_prev:
            e_a = -ask_q
        elif ask_p > pa_prev:
            e_a = +qa_prev
        else:
            e_a = -(ask_q - qa_prev)

        # 상태 업데이트
        self.prev_bid = _BestQuote(bid_p, bid_q)
        self.prev_ask = _BestQuote(ask_p, ask_q)

        e_total = float(e_b + e_a)
        self.last_e_total = e_total

        # -------------------------
        # Rolling window 유지
        # -------------------------
        self.impact_window.append((ts, e_total))
        cutoff = ts - pd.Timedelta(minutes=self.window_minutes)
        while self.impact_window and self.impact_window[0][0] < cutoff:
            self.impact_window.popleft()

        ofi_raw = float(sum(v for _, v in self.impact_window))
        self.ofi_history.append(ofi_raw)

        z_ofi = self._zscore(ofi_raw)

        # ✅ FeatureStore가 읽을 값 갱신
        self.last_ofi_raw = ofi_raw
        self.last_z_ofi = z_ofi

    def _zscore(self, ofi_raw: float) -> float:
        hist = np.asarray(self.ofi_history, dtype="float64")
        if len(hist) < 2:
            return 0.0
        mu = float(hist.mean())
        sigma = float(hist.std(ddof=0))
        return 0.0 if sigma == 0 else float((ofi_raw - mu) / sigma)


# ======================================================
# D. QR (bookTicker best qty imbalance)
# ======================================================

def compute_qr_from_bookticker(tick) -> float:
    """
    QR = (bid_qty - ask_qty) / (bid_qty + ask_qty)
    tick: bookTicker row
    """
    if tick is None:
        return 0.0

    if isinstance(tick, (dict, pd.Series)):
        bq = float(tick.get("best_bid_qty", 0.0))
        aq = float(tick.get("best_ask_qty", 0.0))
    else:
        bq = float(getattr(tick, "best_bid_qty", 0.0))
        aq = float(getattr(tick, "best_ask_qty", 0.0))

    denom = bq + aq
    if denom <= 0:
        return 0.0
    return float((bq - aq) / denom)


# ======================================================
# Extra: close z-score (1m close)
# ======================================================

def compute_close_zscore(klines_1m: pd.DataFrame, window: int = 60) -> dict:
    """
    required: ["open_ts", "close"]
    최신 데이터가 뒤에 있어야 함
    """
    if klines_1m is None or klines_1m.empty or "close" not in klines_1m.columns:
        return {"close": np.nan, "z_close": np.nan, "cdf": np.nan}

    s = pd.to_numeric(klines_1m["close"], errors="coerce").astype("float64")
    cur = float(s.iloc[-1])

    if len(s) < window + 1:
        return {"close": cur, "z_close": np.nan, "cdf": np.nan}

    hist = s.iloc[-window - 1:-1]
    mu = float(hist.mean())
    sigma = float(hist.std(ddof=0))

    z = 0.0 if sigma == 0 else (cur - mu) / sigma
    cdf = float(norm.cdf(z))
    return {"close": cur, "z_close": float(z), "cdf": cdf}

