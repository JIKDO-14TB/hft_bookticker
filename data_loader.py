# hft_backtest_engine/data_loader.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, List

import numpy as np
import pandas as pd


# =========================
# Config
# =========================
UTC_TZ = "UTC"


@dataclass(frozen=True)
class DataPaths:
    root: Path  # .../FPA_crypto_project/data
    agg_trades_dir: str = "aggTrades"
    book_depth_dir: str = "bookDepth"
    funding_dir: str = "funding_rate_daily"
    klines_1m_dir: str = "klines_1m"

    def agg_day(self, symbol: str, ymd: str) -> Path:
        return self.root / self.agg_trades_dir / symbol / f"{ymd}.parquet"

    def book_day(self, symbol: str, ymd: str) -> Path:
        return self.root / self.book_depth_dir / symbol / f"{ymd}.parquet"

    def funding_dir_path(self, symbol: str) -> Path:
        return self.root / self.funding_dir / symbol

    def klines_1m_day(self, symbol: str, ymd: str) -> Path:
        return self.root / self.klines_1m_dir / symbol / f"{symbol}_{ymd}_1m.parquet"


# =========================
# Helpers: time conversion
# =========================

def _ms_to_utc_datetime(ms: pd.Series) -> pd.Series:
    """
    epoch ms(int64) -> datetime64[ns, UTC]
    """
    dt = pd.to_datetime(ms.astype("int64"), unit="ms", utc=True)
    return dt  # tz-aware UTC


def _ensure_utc_tz(dt: pd.Series) -> pd.Series:
    """
    dt가 naive면 UTC로 localize, tz-aware면 UTC로 convert.
    """
    dt = pd.to_datetime(dt, errors="coerce")
    if getattr(dt.dtype, "tz", None) is None:
        return dt.dt.tz_localize(UTC_TZ)
    return dt.dt.tz_convert(UTC_TZ)


def _coerce_float64(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")


def _coerce_int64(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.isna().any():
                df[c] = s.astype("Int64")
            else:
                df[c] = s.astype("int64")


# =========================
# Loader main
# =========================

class DataLoader:
    def __init__(self, data_root: str | Path):
        self.paths = DataPaths(root=Path(data_root))

    # ---------- discovery ----------
    def list_symbols(self) -> List[str]:
        symbols = set()
        for sub in [
            self.paths.agg_trades_dir,
            self.paths.book_depth_dir,
            self.paths.funding_dir,
            self.paths.klines_1m_dir,
        ]:
            p = self.paths.root / sub
            if p.exists():
                for d in p.iterdir():
                    if d.is_dir():
                        symbols.add(d.name)
        return sorted(symbols)

    def list_available_days(self, symbol: str, dataset: str) -> List[str]:
        """
        dataset: "aggTrades" | "bookDepth" | "funding_rate_daily" | "klines_1m"
        return: ["2025-09-17", ...]
        """
        base = self.paths.root / dataset / symbol
        if not base.exists():
            return []
        days: List[str] = []
        for f in base.glob("*.parquet"):
            name = f.stem
            if dataset in ("aggTrades", "bookDepth"):
                days.append(name)
            else:
                parts = name.split("_")
                for part in parts:
                    if len(part) == 10 and part[4] == "-" and part[7] == "-":
                        days.append(part)
                        break
        return sorted(set(days))

    # ---------- load: aggTrades ----------
    def load_aggtrades_day(self, symbol: str, ymd: str) -> pd.DataFrame:
        fp = self.paths.agg_day(symbol, ymd)
        if not fp.exists():
            raise FileNotFoundError(fp)

        df = pd.read_parquet(fp).copy()

        required = ["agg_trade_id", "price", "quantity", "transact_time", "is_buyer_maker"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"[aggTrades] missing columns: {missing} in {fp}")

        _coerce_int64(df, ["agg_trade_id", "first_trade_id", "last_trade_id"])
        _coerce_float64(df, ["price", "quantity"])

        # 핵심 시간축: UTC
        df["ts"] = _ms_to_utc_datetime(df["transact_time"])

        df["symbol"] = symbol
        df["dtype"] = "aggTrades"
        df = df.sort_values("ts").reset_index(drop=True)

        # aggressor side / signed volume (원하면 제거/변경 가능)
        df["aggressor_side"] = np.where(df["is_buyer_maker"].astype(bool), "SELL", "BUY")
        df["signed_qty"] = np.where(df["aggressor_side"].eq("BUY"), df["quantity"], -df["quantity"]).astype("float64")
        #여기서 매수세, 매도세 판단 힌트 얻어가기.
        return df

    # ---------- load: bookDepth ----------
    def load_bookdepth_day(self, symbol: str, ymd: str) -> pd.DataFrame:
        fp = self.paths.book_day(symbol, ymd)
        if not fp.exists():
            raise FileNotFoundError(fp)

        df = pd.read_parquet(fp).copy()

        required = ["timestamp", "percentage", "depth", "notional"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"[bookDepth] missing columns: {missing} in {fp}")

        # timestamp 문자열은 UTC 문자열이라고 가정
        df["ts"] = _ensure_utc_tz(df["timestamp"])

        _coerce_int64(df, ["percentage"])
        _coerce_float64(df, ["depth", "notional"])

        df["symbol"] = symbol
        df["dtype"] = "bookDepth"
        df = df.sort_values(["ts", "percentage"]).reset_index(drop=True)

        return df

    # ---------- load: funding ----------
    def load_funding_day(self, symbol: str, ymd: str) -> pd.DataFrame:
        base = self.paths.funding_dir_path(symbol)
        if not base.exists():
            raise FileNotFoundError(base)

        candidates = list(base.glob(f"*{ymd}*.parquet"))
        if not candidates:
            raise FileNotFoundError(f"No funding parquet for {symbol} {ymd} under {base}")

        fp = candidates[0]
        df = pd.read_parquet(fp).copy()

        required = ["fundingTime", "fundingRate", "markPrice"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"[funding] missing columns: {missing} in {fp}")

        # fundingTime_kst가 있어도, 여기서는 'UTC 통일'이 목표라 fundingTime(ms)로 UTC 생성이 안전
        df["ts"] = _ms_to_utc_datetime(df["fundingTime"])

        _coerce_float64(df, ["fundingRate", "markPrice"])

        df["symbol"] = symbol
        df["dtype"] = "funding"
        df = df.sort_values("ts").reset_index(drop=True)

        return df

    # ---------- load: klines_1m ----------
    def load_klines_1m_day(self, symbol: str, ymd: str) -> pd.DataFrame:
        fp = self.paths.klines_1m_day(symbol, ymd)
        if not fp.exists():
            raise FileNotFoundError(fp)

        df = pd.read_parquet(fp).copy()

        required = ["open_time_ms", "open", "high", "low", "close", "volume", "close_time_ms"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"[klines_1m] missing columns: {missing} in {fp}")

        df["open_ts"] = _ms_to_utc_datetime(df["open_time_ms"])
        df["close_ts"] = _ms_to_utc_datetime(df["close_time_ms"])

        _coerce_float64(df, ["open", "high", "low", "close", "quote_volume", "taker_buy_base", "taker_buy_quote"])

        # volume은 int로 와도 계산 편하게 float64로 통일
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("float64")
        _coerce_int64(df, ["trades"])

        df["symbol"] = symbol
        df["dtype"] = "klines_1m"
        df = df.sort_values("open_ts").reset_index(drop=True)

        return df

    # =========================
    # Join / Feature helpers
    # =========================

    def attach_bookdepth_asof(
        self,
        trades: pd.DataFrame,
        book: pd.DataFrame,
        tolerance: Optional[pd.Timedelta] = pd.Timedelta("2s"),
    ) -> pd.DataFrame:
        """
        aggTrades의 ts(UTC)에 대해, bookDepth의 가장 최근 스냅샷 피처를 '가까울 때만' 붙인다.
        - tolerance가 None이면 무제한으로 과거 스냅샷을 붙임(비추천)
        """
        if trades is None or len(trades) == 0:
            return trades.copy() if trades is not None else pd.DataFrame()
        if book is None or len(book) == 0:
            out = trades.copy()
            out["book_ts"] = pd.NaT
            return out

        t = trades.sort_values("ts").copy()
        b = book.sort_values("ts").copy()

        # 그대로 붙이면 행 폭증 → ts 스냅샷 단위로 요약한 뒤 asof-join
        snap = self._bookdepth_snapshot_features(b)

        merged = pd.merge_asof(
            t,
            snap.sort_values("ts"),
            on="ts",
            direction="backward",
            tolerance=tolerance,
        )
        return merged

    def _bookdepth_snapshot_features(self, book: pd.DataFrame) -> pd.DataFrame:
        """
        bookDepth(percentage별 depth/notional)를 스냅샷(ts) 단위로 요약.
        percentage 단위(bp/% 등)는 아직 확정 전이므로, 부호로 bid/ask 구분만 함.
        """
        b = book.copy()
        b["side"] = np.where(b["percentage"] < 0, "bid", "ask")

        # 가벼운 기본 피처(원하면 여기서 확장)
        g = b.groupby(["ts", "side"], as_index=False)[["notional", "depth"]].sum()

        piv_n = g.pivot(index="ts", columns="side", values="notional").reset_index()
        piv_d = g.pivot(index="ts", columns="side", values="depth").reset_index()

        piv_n = piv_n.rename(columns={"bid": "book_notional_bid", "ask": "book_notional_ask"})
        piv_d = piv_d.rename(columns={"bid": "book_depth_bid", "ask": "book_depth_ask"})

        out = piv_n.merge(piv_d, on="ts", how="outer").sort_values("ts").reset_index(drop=True)

        # imbalance (총 notional 기반)
        bid = out.get("book_notional_bid")
        ask = out.get("book_notional_ask")
        if bid is not None and ask is not None:
            out["book_imbalance"] = (bid - ask) / (bid + ask + 1e-12)
        else:
            out["book_imbalance"] = np.nan

        out["book_ts"] = out["ts"]
        return out


# =========================
# Quick usage (optional)
# =========================
if __name__ == "__main__":
    loader = DataLoader(r"C:\Users\김도훈\OneDrive\바탕 화면\FPA_crypto_project\data")

    symbol = "0GUSDT"
    day = "2025-09-17"

    trades = loader.load_aggtrades_day(symbol, day)
    book = loader.load_bookdepth_day(symbol, day)

    merged = loader.attach_bookdepth_asof(trades, book, tolerance=pd.Timedelta("2s"))

    print(trades[["transact_time", "ts"]].head())
    print(merged.filter(regex="^(ts|price|quantity|book_)").head())
