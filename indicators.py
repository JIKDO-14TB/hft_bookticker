# hft_backtest_engine/indicators.py
from __future__ import annotations

import numpy as np
import pandas as pd


# ======================================================
# 기본 통계 / 롤링
# ======================================================

def rolling_mean(x: pd.Series, window: int) -> float:
    """최근 window 평균"""
    if len(x) < window:
        return np.nan
    return x.iloc[-window:].mean()


def rolling_std(x: pd.Series, window: int) -> float:
    """최근 window 표준편차"""
    if len(x) < window:
        return np.nan
    return x.iloc[-window:].std(ddof=0)


def zscore(x: pd.Series, window: int) -> float:
    """최근 값의 z-score"""
    if len(x) < window:
        return np.nan

    mean = x.iloc[-window:].mean()
    std = x.iloc[-window:].std(ddof=0)

    if std == 0:
        return 0.0

    return (x.iloc[-1] - mean) / std


# ======================================================
# EMA / 추세
# ======================================================

def ema(x: pd.Series, span: int) -> float:
    """EMA 마지막 값"""
    if len(x) < span:
        return np.nan
    return x.ewm(span=span, adjust=False).mean().iloc[-1]


def ema_slope(x: pd.Series, span: int) -> float:
    """EMA 기울기 (최근 2개 차이)"""
    if len(x) < span + 1:
        return np.nan
    ema_series = x.ewm(span=span, adjust=False).mean()
    return ema_series.iloc[-1] - ema_series.iloc[-2]


# ======================================================
# 변동성
# ======================================================

def realized_volatility(returns: pd.Series, window: int) -> float:
    """
    실현 변동성 (sqrt(sum(r^2)))
    returns: log return or simple return series
    """
    if len(returns) < window:
        return np.nan
    return np.sqrt(np.sum(returns.iloc[-window:] ** 2))


# ======================================================
# Trade 기반 지표
# ======================================================

def signed_volume(
    prices: pd.Series,
    sizes: pd.Series,
    sides: pd.Series,
    window: int,
) -> float:
    """
    signed volume
    sides: +1 (aggressive buy), -1 (aggressive sell)
    """
    if len(prices) < window:
        return np.nan

    sv = sizes.iloc[-window:] * sides.iloc[-window:]
    return sv.sum()


def trade_intensity(timestamps: pd.Series, window_seconds: int) -> float:
    """
    최근 window_seconds 동안 체결 빈도
    """
    if len(timestamps) < 2:
        return 0.0

    cutoff = timestamps.iloc[-1] - pd.Timedelta(seconds=window_seconds)
    return (timestamps >= cutoff).sum()


# ======================================================
# OLS (간단 버전)
# ======================================================

def ols_slope(y: pd.Series) -> float:
    """
    단순 OLS slope (y ~ t)
    """
    n = len(y)
    if n < 2:
        return np.nan

    x = np.arange(n)
    x_mean = x.mean()
    y_mean = y.mean()

    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0

    return ((x - x_mean) * (y - y_mean)).sum() / denom


# ======================================================
# 안전 보조 함수
# ======================================================

def clip(x: float, lo: float, hi: float) -> float:
    """값 클리핑"""
    return max(lo, min(hi, x))


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    """0 division 방지"""
    if b == 0:
        return default
    return a / b
