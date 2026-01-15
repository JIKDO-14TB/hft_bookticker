# hft_backtest_engine/reporting.py
from __future__ import annotations

import pandas as pd


def build_trade_dfs(engine):
    fills_df = pd.DataFrame(engine.fills)
    closed_df = pd.DataFrame(engine.closed_trades)

    if not fills_df.empty:
        fills_df = fills_df.sort_values("ts").reset_index(drop=True)

    if not closed_df.empty:
        closed_df = closed_df.sort_values("exit_ts").reset_index(drop=True)

    return fills_df, closed_df


def summarize_trades(closed_df: pd.DataFrame) -> dict:
    if closed_df.empty:
        return {}

    win_rate = float(closed_df["win"].mean())
    avg_holding = float(closed_df["holding_sec"].mean())
    n_trades = int(len(closed_df))

    pnl_by_symbol = (
        closed_df.groupby("symbol")["pnl"]
        .sum()
        .sort_values(ascending=False)
        .to_dict()
    )

    return {
        "Trades": n_trades,
        "Win Rate": win_rate,
        "Avg Holding (sec)": avg_holding,
        "PnL by Symbol": pnl_by_symbol,
    }
