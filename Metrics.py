"""
Evaluation: turnover-aware pnl, trading stats, classification stats.

    r_strategy_t = signal_t * r_t  -  (bps/1e4) * |signal_t - signal_{t-1}|

A flip from -1 to +1 is two units of turnover because it crosses the spread
twice.  Everything is computed from first principles so the numbers can be
checked by hand.
"""

import numpy as np
import pandas as pd

from config import PERIODS_PER_YEAR, COST_SWEEP_BPS


def attach_realised(df):
    """The outcome signal_t was betting on: it is set at close t-1 and earns r_t."""
    out = df.copy()
    out["y_up_realised"] = (out["returns"] > 0).astype(float)
    out.loc[out["returns"].isna(), "y_up_realised"] = np.nan
    return out


def apply_costs(df, bps=0.0):
    out = df.copy()
    sig, ret = out["signal"].fillna(0.0), out["returns"].fillna(0.0)
    out["turnover"] = sig.diff().abs().fillna(sig.abs())
    out["gross_returns"] = sig * ret
    out["cost"] = (bps / 1e4) * out["turnover"]
    out["strategy_returns"] = out["gross_returns"] - out["cost"]
    out["bh_cum"] = (1 + ret).cumprod()
    out["gross_cum"] = (1 + out["gross_returns"]).cumprod()
    out["strategy_cum"] = (1 + out["strategy_returns"]).cumprod()
    return out


def perf_stats(returns, ppy=PERIODS_PER_YEAR):
    r = pd.Series(returns).dropna()
    if r.empty:
        return {}
    cum = (1 + r).cumprod()
    years = len(r) / ppy
    cagr = float(cum.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan
    mdd = float((cum / cum.cummax() - 1).min())
    down = r[r < 0]
    gains, losses = r[r > 0].sum(), -down.sum()
    sd = r.std(ddof=1)
    return {
        "Cumulative return %": float(cum.iloc[-1] - 1) * 100,
        "CAGR %": cagr * 100,
        "Annual vol %": float(sd * np.sqrt(ppy)) * 100,
        "Sharpe": float(r.mean() / sd * np.sqrt(ppy)) if sd > 0 else np.nan,
        "Sortino": float(r.mean() * ppy / (down.std(ddof=1) * np.sqrt(ppy)))
                   if len(down) > 1 and down.std(ddof=1) > 0 else np.nan,
        "Max drawdown %": mdd * 100,
        "Calmar": float(cagr / abs(mdd)) if mdd < 0 else np.nan,
        "Win rate %": float((r > 0).mean() * 100),
        "Profit factor": float(gains / losses) if losses > 0 else np.nan,
    }


def trade_stats(df, ppy=PERIODS_PER_YEAR):
    sig = df["signal"].fillna(0.0)
    turn = df.get("turnover", sig.diff().abs().fillna(sig.abs()))
    changes = int((sig != sig.shift(1)).sum())
    return {
        "Bars": len(sig),
        "Time in market %": float((sig != 0).mean() * 100),
        "Long %": float((sig > 0).mean() * 100),
        "Short %": float((sig < 0).mean() * 100),
        "Position changes": changes,
        "Avg holding (bars)": float(len(sig) / changes) if changes else np.nan,
        "Annualised turnover": float(turn.mean() * ppy),
    }


def classification_stats(df):
    """Is the probability itself any good, independent of position sizing?"""
    d = df[["prob_up", "y_up_realised", "signal"]].dropna()
    if d.empty:
        return {}
    y, p = d["y_up_realised"].astype(int).to_numpy(), d["prob_up"].to_numpy()
    pred = (p > 0.5).astype(int)
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan
    except Exception:
        auc = np.nan
    traded = d[d["signal"] != 0]
    return {
        "N predictions": len(d),
        "Base rate (up) %": float(y.mean() * 100),
        "Accuracy %": float((pred == y).mean() * 100),
        "ROC-AUC": auc,
        "Brier score": float(np.mean((p - y) ** 2)),
        "Brier (base rate)": float(np.mean((y.mean() - y) ** 2)),
        "Hit rate on traded bars %": float(
            ((traded["prob_up"] > 0.5).astype(int) ==
             traded["y_up_realised"].astype(int)).mean() * 100) if len(traded) else np.nan,
    }


def cost_sweep(df, bps_list=None, ppy=PERIODS_PER_YEAR):
    rows = []
    for b in (bps_list if bps_list is not None else COST_SWEEP_BPS):
        st = perf_stats(apply_costs(df, b)["strategy_returns"], ppy)
        rows.append({"cost_bps": b, "Sharpe": st.get("Sharpe"),
                     "CAGR %": st.get("CAGR %"),
                     "Max drawdown %": st.get("Max drawdown %")})
    return pd.DataFrame(rows).set_index("cost_bps")


def breakeven_cost_bps(df):
    """Cost at which net return hits zero: the edge per unit of turnover."""
    gross = (df["signal"].fillna(0) * df["returns"].fillna(0)).mean()
    turn = df["signal"].fillna(0).diff().abs().fillna(0).mean()
    return float(gross / turn * 1e4) if turn > 0 else np.nan


def delay_sensitivity(df, delays=(0, 1, 2), bps=0.0, ppy=PERIODS_PER_YEAR):
    """
    Leakage smoke test: execute the same signal d bars later.  A real 1-bar
    edge decays smoothly; a timing bug collapses instantly.
    """
    rows = []
    for d in delays:
        tmp = df.copy()
        tmp["signal"] = tmp["signal"].shift(d)
        st = perf_stats(apply_costs(tmp, bps)["strategy_returns"], ppy)
        rows.append({"delay_bars": d, "Sharpe": st.get("Sharpe"), "CAGR %": st.get("CAGR %")})
    return pd.DataFrame(rows).set_index("delay_bars")


def summary_table(df, bps=0.0, ppy=PERIODS_PER_YEAR):
    r = apply_costs(df, bps)
    return pd.DataFrame({
        "Buy & Hold": perf_stats(r["returns"], ppy),
        "Strategy (gross)": perf_stats(r["gross_returns"], ppy),
        f"Strategy (net {bps:g}bps)": perf_stats(r["strategy_returns"], ppy),
    }).round(3)
