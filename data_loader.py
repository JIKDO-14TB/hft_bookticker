# hft_backtest_engine/data_loader.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

UTC_TZ = "UTC"


# =====================================================
# Paths
# =====================================================

@dataclass(frozen=True)
class DataPaths:
    root: Path
    agg_trades_dir: str = "um_daily/aggTrades"
    book_ticker_dir: str = "um_daily/bookTicker"
    klines_1m_dir: str = "klines_1m"

    def agg_day(self, symbol: str, ymd: str) -> Path:
        return self.root / self.agg_trades_dir / symbol / f"{ymd}.parquet"

    def bookticker_day(self, symbol: str, ymd: str) -> Path:
        return self.root / self.book_ticker_dir / symbol / f"{ymd}.parquet"

    def klines_1m_day(self, symbol: str, ymd: str) -> Path:
        return self.root / self.klines_1m_dir / symbol / f"{symbol}_{ymd}_1m.parquet"


# =====================================================
# Helpers
# =====================================================

def _ms_to_utc_datetime(ms: pd.Series) -> pd.Series:
    s = pd.to_numeric(ms, errors="coerce")
    # Int64로 두면 pd.to_datetime에서 종종 느려질 수 있어 int64로 강제
    s = s.fillna(0).astype("int64")
    return pd.to_datetime(s, unit="ms", utc=True)


def _coerce_float64(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")


def _coerce_int64(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")


# =====================================================
# DataLoader
# =====================================================

class DataLoader:
    def __init__(self, data_root: str | Path):
        self.paths = DataPaths(root=Path(data_root))

    # -------------------------
    # aggTrades (VPIN용)
    # -------------------------
    def load_aggtrades_day(self, symbol: str, ymd: str) -> pd.DataFrame:
        fp = self.paths.agg_day(symbol, ymd)
        if not fp.exists():
            raise FileNotFoundError(fp)

        df = pd.read_parquet(fp).copy()

        required = ["agg_trade_id", "price", "quantity", "transact_time", "is_buyer_maker"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"[aggTrades] missing columns: {missing} in {fp}")

        _coerce_int64(df, ["agg_trade_id", "first_trade_id", "last_trade_id", "transact_time"])
        _coerce_float64(df, ["price", "quantity"])

        df["ts"] = _ms_to_utc_datetime(df["transact_time"])
        df["symbol"] = symbol
        df["dtype"] = "aggTrades"

        # (선택) 혹시 bool이 문자열로 들어온 경우 대비
        if df["is_buyer_maker"].dtype == "object":
            df["is_buyer_maker"] = df["is_buyer_maker"].astype(str).str.upper().isin(["TRUE", "1", "T", "Y"])

        return df.sort_values("ts").reset_index(drop=True)

    # -------------------------
    # bookTicker (Hybrid clock 집계)
    # -------------------------
    def load_bookticker_day(
        self,
        symbol: str,
        ymd: str,
        clock_freq: Optional[str] = "100ms",
        drop_partial_last_bin: bool = False,
    ) -> pd.DataFrame:
        """
        bookTicker를 하이브리드 클럭으로 집계해서 반환.

        반환 컬럼(핵심)
        - ts : clock bin timestamp (UTC tz-aware)
        - best_bid_price_min / best_bid_price_max
        - best_ask_price_min / best_ask_price_max
        - best_bid_qty (last) / best_ask_qty (last)
        - price : mid(last)  = 0.5*(bid_last + ask_last)

        clock_freq=None 이면 원본 tick 그대로 반환(기존 방식 호환)
        """
        fp = self.paths.bookticker_day(symbol, ymd)
        if not fp.exists():
            raise FileNotFoundError(fp)

        df = pd.read_parquet(fp).copy()

        required = [
            "update_id",
            "best_bid_price",
            "best_bid_qty",
            "best_ask_price",
            "best_ask_qty",
            "transaction_time",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"[bookTicker] missing columns: {missing} in {fp}")

        _coerce_int64(df, ["update_id", "transaction_time"])
        _coerce_float64(df, ["best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty"])

        # ✅ 공통 시간축: transaction_time(ms) -> ts
        df["ts"] = _ms_to_utc_datetime(df["transaction_time"])

        # ✅ 체결 안정성 필터 (원본에서도 적용)
        df = df[(df["best_bid_price"] > 0) & (df["best_ask_price"] > 0)]
        df = df[df["best_bid_price"] <= df["best_ask_price"]]

        df = df.sort_values("ts").reset_index(drop=True)

        # -------------------------
        # (A) 원본 tick 그대로 반환 (디버그/비교용)
        # -------------------------
        if clock_freq is None:
            df["price"] = 0.5 * (df["best_bid_price"] + df["best_ask_price"])
            df["symbol"] = symbol
            df["dtype"] = "bookTicker"
            return df

        # -------------------------
        # (B) Hybrid clock 집계
        # -------------------------
        # 인덱스 기반 grouper가 가장 빠르고 안전
        dfi = df.set_index("ts", drop=False)

        g = dfi.groupby(pd.Grouper(key="ts", freq=clock_freq))

        agg = g.agg(
            best_bid_price_min=("best_bid_price", "min"),
            best_bid_price_max=("best_bid_price", "max"),
            best_ask_price_min=("best_ask_price", "min"),
            best_ask_price_max=("best_ask_price", "max"),
            best_bid_price_last=("best_bid_price", "last"),
            best_ask_price_last=("best_ask_price", "last"),
            best_bid_qty=("best_bid_qty", "last"),
            best_ask_qty=("best_ask_qty", "last"),
            n_ticks=("update_id", "count"),
            transaction_time_last=("transaction_time", "last"),
            update_id_last=("update_id", "last"),
        )

        # 빈 bin 제거
        agg = agg.dropna(subset=["best_bid_price_last", "best_ask_price_last"]).reset_index()

        if drop_partial_last_bin and len(agg) > 1:
            # 마지막 bin은 데이터가 덜 들어갈 수 있어(선택)
            agg = agg.iloc[:-1].copy()

        # ✅ mid-price는 last 기준 (Strategy / 로그용)
        agg["price"] = 0.5 * (agg["best_bid_price_last"] + agg["best_ask_price_last"])

        # dtype/심볼
        agg["symbol"] = symbol
        agg["dtype"] = f"bookTicker_{clock_freq}"

        # “기존 코드 호환”을 위해 last를 기본 필드명으로도 제공(선택)
        # -> Execution/Strategy가 best_bid_price를 찾는다면 이걸 쓰게 됨
        agg["best_bid_price"] = agg["best_bid_price_last"]
        agg["best_ask_price"] = agg["best_ask_price_last"]

        # 정렬
        agg = agg.sort_values("ts").reset_index(drop=True)
        return agg

    # -------------------------
    # klines 1m
    # -------------------------
    def load_klines_1m_day(self, symbol: str, ymd: str) -> pd.DataFrame:
        fp = self.paths.klines_1m_day(symbol, ymd)
        if not fp.exists():
            raise FileNotFoundError(fp)

        df = pd.read_parquet(fp).copy()

        required = ["open_time_ms", "open", "high", "low", "close", "volume", "close_time_ms"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"[klines_1m] missing columns: {missing} in {fp}")

        _coerce_int64(df, ["open_time_ms", "close_time_ms"])
        df["open_ts"] = _ms_to_utc_datetime(df["open_time_ms"])
        df["close_ts"] = _ms_to_utc_datetime(df["close_time_ms"])

        _coerce_float64(df, ["open", "high", "low", "close", "volume"])
        if "trades" in df.columns:
            _coerce_int64(df, ["trades"])

        df["symbol"] = symbol
        df["dtype"] = "klines_1m"

        return df.sort_values("open_ts").reset_index(drop=True)
#100ms 로 다운샘플링. 그 사이 book ticker 분봉처럼 만들어서 체결 판단할거임.