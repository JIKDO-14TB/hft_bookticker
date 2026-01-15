import pandas as pd
from hft_backtest_engine.data_loader import DataLoader


def estimate_vpin_bucket_volume(
    loader: DataLoader,
    symbol: str,
    day: str,
    target_buckets: int,
) -> float:
    """
    하루 aggTrades 기준으로 VPIN bucket volume 자동 산정

    target_buckets:
      - 하루 동안 만들고 싶은 VPIN bucket 개수
      - 예: 500 ~ 1500 (BTC 기준)
    """
    df = loader.load_aggtrades_day(symbol, day)

    if df is None or df.empty:
        raise ValueError(f"No aggTrades for {symbol} {day}")

    total_qty = float(df["quantity"].sum())
    bucket_volume = total_qty / float(target_buckets)

    return float(bucket_volume)