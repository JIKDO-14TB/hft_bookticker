# hft_backtest_engine/metrics.py
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def compute_metrics(
    equity_curve: pd.DataFrame,
    fills: pd.DataFrame | None = None,
    initial_capital: float = 100.0,
    freq_per_year: int = 252 * 24 * 60,
) -> dict:
    """
    equity_curve columns:
    - ts
    - equity
    - cash
    - n_positions

    fills (optional):
    - ts
    - fill_type ("ENTER"/"EXIT")
    - size (notional)
    - fee_amt or fee_cash
    - gross_pnl / net_pnl (보통 EXIT row에만 존재)
    """

    if equity_curve is None or equity_curve.empty:
        return {}

    eq = equity_curve.copy().sort_values("ts").reset_index(drop=True)

    # =========================
    # 1) 단리 PnL / Return
    # =========================
    eq["pnl"] = eq["equity"] - float(initial_capital)
    eq["ret"] = eq["equity"].diff().fillna(0.0)

    mean_ret = float(eq["ret"].mean())
    std_ret = float(eq["ret"].std(ddof=0))

    sharpe = (mean_ret / (std_ret + 1e-12) * np.sqrt(freq_per_year)) if std_ret > 0 else 0.0

    # =========================
    # 2) MDD (cash)
    # =========================
    cummax = eq["equity"].cummax()
    dd = eq["equity"] - cummax
    mdd = float(dd.min())  # 음수

    total_pnl = float(eq["pnl"].iloc[-1])
    total_ret_pct = 100.0 * total_pnl / float(initial_capital)
    rmdd = (total_pnl / abs(mdd)) if mdd < 0 else np.nan

    out = {
        "Final Equity": float(eq["equity"].iloc[-1]),
        "Total PnL (equity)": total_pnl,
        "Total Return (%)": float(total_ret_pct),
        "Sharpe": float(sharpe),
        "MDD (%)": float(100.0 * mdd / float(initial_capital)),
        "RMDD": float(rmdd) if np.isfinite(rmdd) else np.nan,
        "Mean Ret": mean_ret,
        "Std Ret": std_ret,
    }

    # =========================
    # 3) fills 기반: Turnover / Fees / Gross vs Net
    # =========================
    if fills is not None and not fills.empty:
        f = fills.copy()

        # --- Turnover (%) ---
        if "size" in f.columns:
            traded_notional = float(pd.to_numeric(f["size"], errors="coerce").fillna(0.0).sum())
            out["Turnover (%)"] = float(100.0 * traded_notional / float(initial_capital))

        # --- Total Fees ---
        fee_col = None
        if "fee_amt" in f.columns:
            fee_col = "fee_amt"
        elif "fee_cash" in f.columns:
            fee_col = "fee_cash"

        if fee_col is not None:
            total_fees = float(pd.to_numeric(f[fee_col], errors="coerce").fillna(0.0).sum())
            out["Total Fees (cash)"] = total_fees

        # --- Gross vs Net PnL (EXIT row sum) ---
        # fills에 gross_pnl/net_pnl이 있다면 EXIT만 집계해서 보여줌
        if "fill_type" in f.columns:
            f_exit = f[f["fill_type"].astype(str).str.upper().eq("EXIT")].copy()
        else:
            f_exit = f.copy()

        if ("gross_pnl" in f_exit.columns) and ("net_pnl" in f_exit.columns):
            gross_pnl = float(pd.to_numeric(f_exit["gross_pnl"], errors="coerce").fillna(0.0).sum())
            net_pnl = float(pd.to_numeric(f_exit["net_pnl"], errors="coerce").fillna(0.0).sum())
            out["Gross PnL (sum EXIT)"] = gross_pnl
            out["Net PnL (sum EXIT)"] = net_pnl
            out["Gross Return (%)"] = float(100.0 * gross_pnl / float(initial_capital))
            out["Net Return (%)"] = float(100.0 * net_pnl / float(initial_capital))

    return out


def plot_cum_pnl(
    equity_curve: pd.DataFrame,
    initial_capital: float = 100.0,
    title: str = "Cumulative PnL (Simple)",
):
    """
    단리 기준 누적 PnL 곡선 (기간 표시 포함)
    """
    if equity_curve is None or equity_curve.empty:
        print("[WARN] equity_curve is empty.")
        return

    eq = equity_curve.sort_values("ts").copy()
    eq["cum_pnl"] = eq["equity"] - float(initial_capital)

    start = pd.to_datetime(eq["ts"].iloc[0])
    end = pd.to_datetime(eq["ts"].iloc[-1])

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(eq["ts"], eq["cum_pnl"], linewidth=2)
    ax.axhline(0.0, linestyle="--", alpha=0.6)

    ax.set_title(title)
    ax.set_xlabel(f"Time  ({start} ~ {end})")  # ✅ 기간 표시
    ax.set_ylabel("PnL (cash, simple)")

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()



