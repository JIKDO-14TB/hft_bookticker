from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm


# ======================================================
# A. VPIN (Stateful, time-decayed)
# ======================================================

class VPINCalculator:
    """
    VPIN bucket을 tick 단위로 누적하면서 상태 유지.
    - bucket은 완성 시점 기준 4분 59초 후 자동 폐기
    """

    def __init__(
        self,
        bucket_volume: float,
        history: int = 288,              # ✅ 히스토리도 288로
        bucket_ttl_seconds: int = 299,   # ✅ 4분 59초
    ):
        self.bucket_volume = float(bucket_volume)
        self.history = int(history)
        self.bucket_ttl = pd.Timedelta(seconds=bucket_ttl_seconds)

        self.acc_vol = 0.0
        self.buy_vol = 0.0
        self.sell_vol = 0.0

        # (completed_ts, imbalance)
        self.bucket_imbalances: Deque[Tuple[pd.Timestamp, float]] = deque()

    # --------------------------------------------------
    # Tick update
    # --------------------------------------------------
    def update_trade(self, trade) -> None:
        """
        trade: aggTrades itertuples row
        required fields: ts, quantity, is_buyer_maker
        """
        q = float(trade.quantity)
        self.acc_vol += q

        if bool(trade.is_buyer_maker):
            self.sell_vol += q
        else:
            self.buy_vol += q

        # bucket 완성
        if self.acc_vol >= self.bucket_volume:
            denom = max(self.buy_vol + self.sell_vol, 1e-12)
            imb = abs(self.buy_vol - self.sell_vol) / denom

            completed_ts = trade.ts
            self.bucket_imbalances.append((completed_ts, float(imb)))

            # reset
            self.acc_vol = 0.0
            self.buy_vol = 0.0
            self.sell_vol = 0.0

            # history 초과 제거 (개수 기준)
            while len(self.bucket_imbalances) > self.history:
                self.bucket_imbalances.popleft()

    # --------------------------------------------------
    # Value (with TTL decay)
    # --------------------------------------------------
    def get_value(self, now_ts: Optional[pd.Timestamp] = None) -> dict:
        """
        now_ts: 현재 시각 (FeatureStore에서 trade.ts 전달 권장)
        """
        if not self.bucket_imbalances:
            return {"vpin_raw": np.nan, "vpin_cdf": 0.0}

        if now_ts is not None:
            cutoff = now_ts - self.bucket_ttl
            while self.bucket_imbalances and self.bucket_imbalances[0][0] < cutoff:
                self.bucket_imbalances.popleft()

        if not self.bucket_imbalances:
            return {"vpin_raw": np.nan, "vpin_cdf": 0.0}

        # 최신 버킷
        vpin_raw = self.bucket_imbalances[-1][1]

        hist = np.asarray([v for _, v in self.bucket_imbalances], dtype="float64")
        vpin_cdf = float(np.mean(hist <= vpin_raw))

        return {
            "vpin_raw": float(vpin_raw),
            "vpin_cdf": float(vpin_cdf),
        }


# ======================================================
# B. Trade Count Spike (N) - Stateless
# ======================================================
#이 신호가 평소보다 얼마나 중요한가.
def compute_trade_count_spike(
    klines_1m: pd.DataFrame,
    window: int = 60, #현재 1분의 거래횟수가 과거 60분과 비교했을 때 얼마나 이례적인가
) -> dict:
    """
    klines_1m columns required: ["open_ts", "trades"]
    최신 데이터가 뒤에 있어야 함
    """
    if klines_1m is None or klines_1m.empty:
        return {"tc": 0.0, "z_tc": 0.0, "n_cdf": 0.5}

    if "trades" not in klines_1m.columns:
        return {"tc": 0.0, "z_tc": 0.0, "n_cdf": 0.5}

    if len(klines_1m) < window + 1:
        return {"tc": float(klines_1m["trades"].iloc[-1]), "z_tc": 0.0, "n_cdf": 0.5}
                            #활동성의 대리 변수
    tc_series = pd.to_numeric(klines_1m["trades"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    tc_cur = float(tc_series[-1])

    hist = tc_series[-window - 1:-1]
    mu = float(hist.mean())
    sigma = float(hist.std(ddof=0))

    z = 0.0 if sigma == 0 else (tc_cur - mu) / sigma #zscore화
    n_cdf = float(norm.cdf(z)) #0~1사이 스케일의 시장 활성도 계수

    return {"tc": tc_cur, "z_tc": float(z), "n_cdf": n_cdf}


# ======================================================
# C. OFI (Stateful, Spec-accurate, 5min rolling sum + correct z-score)
# ======================================================

@dataclass
class _QuoteState:
    price: float
    depth: float


class OFICalculator:
    """
    스펙 준수 버전:
    - snapshot(ts) 단위로 bid/ask의 proxy price, depth를 만든 다음
    - bid는 bid끼리, ask는 ask끼리 prev와 비교해서 impact 계산
    - impact를 (ts, e_total) deque로 저장해서 최근 window_minutes 합을 ofi_raw로 만듦
    - z_ofi는 ofi_raw 시계열(최근 z_window개)로 계산

    중요:
    - FeatureStore가 10분 버퍼 slice를 계속 넘겨도,
      내부에서 last_processed_ts 이후 snapshot만 처리해서 중복 누적 방지.
    """
    #2~5분동안 실제로 시장을 움직인 힘.
    def __init__(self, window_minutes: int = 5, z_window: int = 30): #5분 동안 발생한 모든 호가장 충격을 누적, 최근 zwindow개 OFI의
                                                                    #이례적인 수준 체크, 30개 OFI 관측치
        self.window_minutes = int(window_minutes)
        self.z_window = int(z_window)

        self.prev_bid: Optional[_QuoteState] = None
        self.prev_ask: Optional[_QuoteState] = None

        self.last_processed_ts: Optional[pd.Timestamp] = None

        # 최근 snapshot impacts (rolling OFI raw 계산용)
        self.impact_window: Deque[Tuple[pd.Timestamp, float]] = deque()

        # ofi_raw의 히스토리(= 5분 누적값의 시계열)
        self.ofi_history: Deque[float] = deque(maxlen=self.z_window)

    def update(self, book: pd.DataFrame) -> dict:
        """
        book: bookDepth slice (ts, percentage, depth, notional 포함)
        - percentage는 -1, +1만 사용
        """
        if book is None or book.empty:
            return {"ofi_raw": 0.0, "z_ofi": 0.0}

        # 필수 컬럼 체크
        need = {"ts", "percentage", "depth", "notional"}
        if not need.issubset(set(book.columns)):
            return {"ofi_raw": 0.0, "z_ofi": 0.0}

        b = book.loc[book["percentage"].isin([-1, 1]), ["ts", "percentage", "depth", "notional"]].copy()
        if b.empty:
            return {"ofi_raw": 0.0, "z_ofi": 0.0}

        # 숫자 안정화
        b["depth"] = pd.to_numeric(b["depth"], errors="coerce").fillna(0.0)
        b["notional"] = pd.to_numeric(b["notional"], errors="coerce").fillna(0.0)

        # snapshot(ts)별로 bid/ask depth/notional 합산
        g = (
            b.groupby(["ts", "percentage"], as_index=False)[["depth", "notional"]]
            .sum()
            .sort_values("ts")
        )

        # pivot -> ts 단위로 bid/ask가 한 줄에 오게
        piv_d = g.pivot(index="ts", columns="percentage", values="depth")
        piv_n = g.pivot(index="ts", columns="percentage", values="notional")

        # 컬럼명: -1, +1
        bid_depth = piv_d.get(-1)
        ask_depth = piv_d.get(1)
        bid_notional = piv_n.get(-1)
        ask_notional = piv_n.get(1)

        # 없는 경우 0
        bid_depth = bid_depth.fillna(0.0) if bid_depth is not None else pd.Series(dtype="float64")
        ask_depth = ask_depth.fillna(0.0) if ask_depth is not None else pd.Series(dtype="float64")
        bid_notional = bid_notional.fillna(0.0) if bid_notional is not None else pd.Series(dtype="float64")
        ask_notional = ask_notional.fillna(0.0) if ask_notional is not None else pd.Series(dtype="float64")

        # proxy price: notional/depth (depth 0 방어)
        bid_price = (bid_notional / (bid_depth.replace(0.0, np.nan))).fillna(0.0)
        ask_price = (ask_notional / (ask_depth.replace(0.0, np.nan))).fillna(0.0)
        #호가창 중심 가격 이동을 감지.
        # 처리할 snapshot만 (중복 방지)
        snap_ts = bid_price.index.union(ask_price.index).sort_values()
        if self.last_processed_ts is not None:
            snap_ts = snap_ts[snap_ts > self.last_processed_ts]

        if len(snap_ts) == 0:
            # 새 snapshot 없으면 현재 rolling ofi_raw만 반환
            ofi_raw = float(sum(v for _, v in self.impact_window))
            z_ofi = self._zscore(ofi_raw)
            return {"ofi_raw": ofi_raw, "z_ofi": z_ofi}

        # snapshot 순회
        for ts in snap_ts:
            pb = float(bid_price.get(ts, 0.0))
            qb = float(bid_depth.get(ts, 0.0))
            pa = float(ask_price.get(ts, 0.0))
            qa = float(ask_depth.get(ts, 0.0))

            # Bid impact (bid끼리 prev 비교)
            e_b = 0.0
            if self.prev_bid is None:
                self.prev_bid = _QuoteState(price=pb, depth=qb)
            else:
                if pb > self.prev_bid.price:
                    e_b = +qb
                elif pb < self.prev_bid.price:
                    e_b = -self.prev_bid.depth
                else:
                    e_b = qb - self.prev_bid.depth
                self.prev_bid = _QuoteState(price=pb, depth=qb)

            # Ask impact (ask끼리 prev 비교)
            e_a = 0.0
            if self.prev_ask is None:
                self.prev_ask = _QuoteState(price=pa, depth=qa)
            else:
                if pa < self.prev_ask.price:
                    e_a = -qa
                elif pa > self.prev_ask.price:
                    e_a = +self.prev_ask.depth
                else:
                    e_a = -(qa - self.prev_ask.depth)
                self.prev_ask = _QuoteState(price=pa, depth=qa)

            e_total = float(e_b + e_a)

            # rolling window에 저장, ask나 bid 중에서 누가 물러났는가를 반영
            self.impact_window.append((ts, e_total))#snapshot 이벤트 기반 누적
            self.last_processed_ts = ts 

            # window_minutes 밖 제거
            cutoff = ts - pd.Timedelta(minutes=self.window_minutes)
            while self.impact_window and self.impact_window[0][0] < cutoff:
                self.impact_window.popleft()

        ofi_raw = float(sum(v for _, v in self.impact_window))

        # z-score는 "ofi_raw 시계열"로
        self.ofi_history.append(ofi_raw)
        z_ofi = self._zscore(ofi_raw)

        return {"ofi_raw": ofi_raw, "z_ofi": z_ofi}

    def _zscore(self, ofi_raw: float) -> float:
        hist = np.asarray(self.ofi_history, dtype="float64")
        if len(hist) < 2:
            return 0.0
        mu = float(hist.mean())
        sigma = float(hist.std(ddof=0))
        return 0.0 if sigma == 0 else float((ofi_raw - mu) / sigma)


# ======================================================
# D. QR (Stateless OK)
# ======================================================
#호가창이 어느 쪽으로 기울어 있는가
def compute_qr(book: pd.DataFrame) -> float:
    """
    book: 최신 snapshot (ts 동일)
    required: percentage, depth
    """
    if book is None or book.empty:
        return 0.0
    if ("percentage" not in book.columns) or ("depth" not in book.columns):
        return 0.0

    bid = pd.to_numeric(book.loc[book["percentage"] == -1, "depth"], errors="coerce").fillna(0.0).sum()
    ask = pd.to_numeric(book.loc[book["percentage"] == 1, "depth"], errors="coerce").fillna(0.0).sum()

    denom = bid + ask
    if denom == 0:
        return 0.0
    return float((bid - ask) / denom) #-1~1사이 값, imbalance 정도


# ======================================================
# Extra: close z-score (1m close)
# ======================================================

def compute_close_zscore(
    klines_1m: pd.DataFrame,
    window: int = 60,
) -> dict:
    """
    klines_1m required: ["open_ts", "close"]
    최신 데이터가 뒤에 있어야 함
    """
    if klines_1m is None or klines_1m.empty:
        return {"close": np.nan, "z_close": np.nan, "cdf": np.nan}

    if "close" not in klines_1m.columns:
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
