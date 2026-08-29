"""
Interpretation: what the model learned and whether to believe it.

Four questions, in order:
  1. FEATURES  what got transformed, and did it become stationary
  2. GATE      what are the states economically, do they persist, and does a
               bull / bear / choppy structure actually appear
  3. EXPERTS   what does each regime's Random Forest use
  4. SIGNAL    is the probability calibrated, and where does the pnl come from
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import PERIODS_PER_YEAR, REPORT_DIR, COST_BPS
from Metrics import apply_costs, perf_stats, cost_sweep, breakeven_cost_bps

COLORS = ["#C1553B", "#B08A3E", "#3B6EA5", "#5A8F5A", "#8A6BAF"]


def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return path


def _state_cols(diag, prefix):
    return [c for c in diag.columns
            if c.startswith(prefix) and c[len(prefix):].isdigit()]


def assigned_state(diag):
    return diag[_state_cols(diag, "p_state_")].to_numpy().argmax(1)


# ---------------------------------------------------------------- 2. GATE
def regime_summary(gate_history):
    """Average fitted parameters per canonical state across all refits."""
    if gate_history.empty:
        return pd.DataFrame()
    g = gate_history.copy(); g["state"] = g.index
    num = [c for c in g.columns if g[c].dtype.kind in "fc" and c != "date"]
    out = g.groupby("state")[num].mean()
    out["label"] = g.groupby("state")["label"].agg(lambda s: s.value_counts().index[0])
    out["label_stability_%"] = g.groupby("state")["label"].agg(
        lambda s: s.value_counts().iloc[0] / len(s) * 100)
    return out


def regime_economics(results, diag, ppy=PERIODS_PER_YEAR):
    """
    Validation, not description: do the states differ in REALISED behaviour
    out of sample?  If the means and vols come out the same, the gate is
    partitioning noise and no amount of labelling will fix that.
    """
    if diag.empty:
        return pd.DataFrame()
    s = pd.Series(assigned_state(diag), index=diag.index, name="state")
    d = pd.concat([s, results["returns"].reindex(s.index)], axis=1).dropna()
    g = d.groupby("state")["returns"]
    out = pd.DataFrame({
        "bars": g.size(),
        "share_%": g.size() / len(d) * 100,
        "mean_ret_daily_%": g.mean() * 100,
        "vol_daily_%": g.std() * 100,
        "ann_return_%": ((1 + g.mean()) ** ppy - 1) * 100,
        "ann_vol_%": g.std() * np.sqrt(ppy) * 100,
        "up_bars_%": g.apply(lambda x: (x > 0).mean() * 100),
    })
    out.index = [f"state_{i}" for i in out.index]
    return out


def persistence_check(gate_history, diag):
    """
    Is this a regime model at all?

    lambda2 is the second eigenvalue of the transition matrix -- the rate at
    which knowing today's state decays into knowing nothing.  Small lambda2
    means A has already converged to its stationary distribution, so P(S_t) is
    near-constant and the architecture quietly degenerates into a fixed-weight
    ensemble of the experts.  That can still make money, but it is not regime
    detection and must not be described as such.
    """
    if gate_history.empty or diag.empty:
        return {}
    l2 = float(gate_history["lambda2"].mean())
    st = pd.Series(assigned_state(diag))
    return {
        "lambda2(A)": l2,
        "state info half-life (bars)": float(
            gate_history["half_life_bars"].replace(np.inf, np.nan).mean()),
        "mean expected duration (bars)": float(
            gate_history["expected_duration_bars"].mean()),
        "realised switches / year": float((st.diff() != 0).mean() * PERIODS_PER_YEAR),
        "mean gate entropy (0=sure, 1=uniform)": float(diag["gate_entropy"].mean()),
        "bars with a >0.7 dominant state %": float(
            (diag[_state_cols(diag, "p_state_")].max(1) > 0.7).mean() * 100),
        "verdict": ("persistent regimes" if l2 >= 0.8 else
                    "NOT regime-like: gate is near-stationary, this behaves as a "
                    "fixed-weight expert ensemble"),
    }


def taxonomy_check(gate_history):
    """
    Did a bull / bear / choppy structure emerge, or did it not?

    Nothing here forces the answer.  Each state is labelled from its own
    fitted numbers -- drift_t = mu*sqrt(duration)/sigma decides bull / bear /
    flat, and the volatility ranking decides low / mid / high.  'Choppy' is
    the name for a state that is flat in drift AND highest in volatility.
    The function simply reports whether such states exist.
    """
    if gate_history.empty:
        return {}
    modal = (gate_history.groupby(gate_history.index)["label"]
             .agg(lambda s: s.value_counts().index[0]))
    labels = list(modal.values)
    has_bull = any(l.startswith("bull") for l in labels)
    has_bear = any(l.startswith("bear") for l in labels)
    choppy = [l for l in labels if l.startswith("flat") and l.endswith("high-vol")]
    out = {f"{k} modal label": v for k, v in modal.items()}
    out["bull state found"] = has_bull
    out["bear state found"] = has_bear
    out["choppy (flat drift + highest vol) found"] = len(choppy) > 0
    out["verdict"] = (
        "bull / bear / choppy structure present" if (has_bull and has_bear and choppy)
        else "the classic three did NOT emerge; the states are what the table says")
    return out


# ---------------------------------------------------------------- 3. EXPERTS
def expert_importance(importances, top_n=15):
    if importances.empty:
        return pd.DataFrame()
    m = importances.groupby(level="state").mean().T
    m.columns = [f"state_{c}" for c in m.columns]
    m["mean"] = m.mean(1)
    return m.sort_values("mean", ascending=False).head(top_n)


def expert_coverage(diag, n_states):
    """How often each expert existed and was actually weighted."""
    if diag.empty:
        return pd.DataFrame()
    rows = {}
    for s in range(n_states):
        rows[f"state_{s}"] = {
            "trained on % of bars": float(diag[f"expert_{s}_trained"].mean() * 100),
            "mean training rows": float(diag[f"n_train_{s}"].dropna().mean()),
            "mean gate weight": float(diag[f"p_state_{s}"].mean()),
            "mean P(up) when weighted": float(diag[f"expert_{s}_prob_up"].mean()),
        }
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------- 4. SIGNAL
def calibration(results, n_bins=10):
    """Predicted P(up) vs realised frequency. A flat curve means no edge."""
    d = results[["prob_up", "y_up_realised", "returns"]].dropna()
    if d.empty:
        return pd.DataFrame()
    edges = np.unique(np.quantile(d["prob_up"], np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return pd.DataFrame()
    g = d.assign(b=pd.cut(d["prob_up"], edges, include_lowest=True)).groupby("b", observed=True)
    return pd.DataFrame({"n": g.size(), "mean_predicted": g["prob_up"].mean(),
                         "realised_up_rate": g["y_up_realised"].mean(),
                         "mean_fwd_return_%": g["returns"].mean() * 100})


def pnl_by_state(results, diag, bps=COST_BPS):
    if diag.empty:
        return pd.DataFrame()
    net = apply_costs(results, bps).loc[diag.index].assign(state=assigned_state(diag))
    g = net.groupby("state")
    out = pd.DataFrame({"bars": g.size(),
                        "gross_pnl_%": g["gross_returns"].sum() * 100,
                        "cost_%": g["cost"].sum() * 100,
                        "net_pnl_%": g["strategy_returns"].sum() * 100,
                        "time_in_market_%": g["signal"].apply(lambda s: (s != 0).mean() * 100)})
    out.index = [f"state_{i}" for i in out.index]
    return out


def pnl_by_side(results, bps=COST_BPS):
    net = apply_costs(results, bps)
    side = pd.cut(net["signal"].fillna(0), [-np.inf, -0.5, 0.5, np.inf],
                  labels=["short", "flat", "long"])
    g = net.groupby(side, observed=True)
    return pd.DataFrame({"bars": g.size(), "gross_pnl_%": g["gross_returns"].sum() * 100,
                         "cost_%": g["cost"].sum() * 100,
                         "net_pnl_%": g["strategy_returns"].sum() * 100,
                         "hit_rate_%": g["gross_returns"].apply(lambda s: (s > 0).mean() * 100)})


# ---------------------------------------------------------------- plots
def plot_regimes(results, diag, gate_history, ticker, outdir=REPORT_DIR):
    if diag.empty:
        return None
    st = pd.Series(assigned_state(diag), index=diag.index)
    px = results["Close"].reindex(st.index)
    labels = ({} if gate_history.empty else
              gate_history.groupby(gate_history.index)["label"]
              .agg(lambda s: s.value_counts().index[0]).to_dict())
    fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                          gridspec_kw={"height_ratios": [2, 1]})
    ax[0].plot(px.index, px.values, color="0.2", lw=1.1, zorder=3)
    for s in sorted(st.unique()):
        ax[0].fill_between(st.index, px.min(), px.max(), where=(st == s).to_numpy(),
                           color=COLORS[s % len(COLORS)], alpha=0.18, step="mid",
                           label=f"state {s}: {labels.get(f'state_{s}', '')}")
    ax[0].set_ylabel("Close"); ax[0].legend(loc="upper left", fontsize=9)
    ax[0].set_title(f"{ticker}: price shaded by inferred regime")
    cols = _state_cols(diag, "p_state_")
    ax[1].stackplot(diag.index, diag[cols].T.to_numpy(), colors=COLORS[:len(cols)],
                    alpha=0.85, labels=cols)
    ax[1].set_ylim(0, 1); ax[1].set_ylabel("P(S_t | info <= t-1)")
    ax[1].legend(loc="upper left", ncol=len(cols), fontsize=8)
    return _save(fig, outdir, "regimes.png")


def plot_importance(importances, outdir=REPORT_DIR, top_n=15):
    imp = expert_importance(importances, top_n)
    if imp.empty:
        return None
    cols = [c for c in imp.columns if c.startswith("state_")]
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(imp) + 2))
    y, h = np.arange(len(imp)), 0.8 / len(cols)
    for i, c in enumerate(cols):
        ax.barh(y + i * h, imp[c].to_numpy(), height=h, label=c, color=COLORS[i % len(COLORS)])
    ax.set_yticks(y + 0.4 - h / 2); ax.set_yticklabels(imp.index); ax.invert_yaxis()
    ax.set_xlabel("mean Gini importance across refits"); ax.legend()
    ax.set_title("What each regime's expert uses")
    return _save(fig, outdir, "importance.png")


def plot_equity(results, bps=COST_BPS, ticker="", outdir=REPORT_DIR):
    net = apply_costs(results, bps)
    fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                          gridspec_kw={"height_ratios": [3, 1]})
    ax[0].plot(net.index, net["bh_cum"], color="0.45", label="Buy & hold")
    ax[0].plot(net.index, net["gross_cum"], color=COLORS[2], ls="--", label="Strategy (gross)")
    ax[0].plot(net.index, net["strategy_cum"], color=COLORS[0],
               label=f"Strategy (net {bps:g}bps)")
    ax[0].set_ylabel("cumulative return"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[0].set_title(f"{ticker}: equity curves")
    dd = net["strategy_cum"] / net["strategy_cum"].cummax() - 1
    ax[1].fill_between(dd.index, dd.values, color=COLORS[0], alpha=.4)
    ax[1].set_ylabel("drawdown"); ax[1].grid(alpha=.3)
    return _save(fig, outdir, "equity.png")


def plot_calibration(results, outdir=REPORT_DIR, n_bins=10):
    t = calibration(results, n_bins)
    if t.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], ls="--", color="0.5", lw=1, label="perfect")
    ax.plot(t["mean_predicted"], t["realised_up_rate"], "o-", color=COLORS[2])
    ax.set_xlabel("predicted P(up)"); ax.set_ylabel("realised up rate")
    ax.set_title("Calibration (out of sample)"); ax.grid(alpha=.3); ax.legend()
    return _save(fig, outdir, "calibration.png")


# ---------------------------------------------------------------- report
def _fmt(obj, title):
    if isinstance(obj, dict):
        if not obj:
            return ""
        body = pd.Series({k: (round(v, 4) if isinstance(v, (int, float, np.floating))
                              and v == v else v) for k, v in obj.items()}).to_string()
    else:
        if obj is None or len(obj) == 0:
            return ""
        try:
            body = obj.round(4).to_string()
        except TypeError:
            body = obj.to_string()
    return f"\n### {title}\n\n```\n{body}\n```\n"


def generate_report(results, artifacts, feat_report=None, ticker="",
                    bps=COST_BPS, outdir=REPORT_DIR, verbose=True):
    os.makedirs(outdir, exist_ok=True)
    diag = artifacts.get("diagnostics", pd.DataFrame())
    imps = artifacts.get("importances", pd.DataFrame())
    gh = artifacts.get("gate_history", pd.DataFrame())
    n_states = artifacts.get("n_states", 0)

    if "y_up_realised" not in results:
        results = results.copy()
        results["y_up_realised"] = (results["returns"] > 0).astype(float)

    s = [f"# Interpretation report -- {ticker}\n",
         f"\nEvaluation: {diag.index.min().date()} to {diag.index.max().date()} "
         f"({len(diag)} decisions), {n_states} states, cost {bps:g} bps.\n"]

    s.append("\n## 1. Features\n")
    if feat_report is not None and len(feat_report):
        s.append(_fmt(feat_report["status"].value_counts().to_frame("n"),
                      "Transform decisions (fitted in-sample, then frozen)"))

    s.append("\n## 2. Regimes\n")
    s.append(_fmt(regime_summary(gh), "Fitted state parameters (mean over refits)"))
    s.append(_fmt(taxonomy_check(gh), "Did bull / bear / choppy emerge?"))
    s.append(_fmt(regime_economics(results, diag),
                  "Realised behaviour per state -- is the split economically real?"))
    s.append(_fmt(persistence_check(gh, diag), "Is this a regime model? (lambda2 of A)"))

    s.append("\n## 3. Experts\n")
    s.append(_fmt(expert_coverage(diag, n_states), "Expert coverage"))
    s.append(_fmt(expert_importance(imps), "Top features per regime"))

    s.append("\n## 4. Signal and pnl\n")
    s.append(_fmt(calibration(results), "Calibration"))
    s.append(_fmt(pnl_by_state(results, diag, bps), "PnL by regime"))
    s.append(_fmt(pnl_by_side(results, bps), "PnL by side"))
    s.append(_fmt(cost_sweep(results), "Cost sensitivity"))
    be = breakeven_cost_bps(results)
    s.append(f"\nBreak-even cost: **{be:.2f} bps** per unit of turnover.\n"
             if be == be else "")

    figs = [f for f in [plot_regimes(results, diag, gh, ticker, outdir),
                        plot_importance(imps, outdir),
                        plot_calibration(results, outdir),
                        plot_equity(results, bps, ticker, outdir)] if f]
    s.append("\n## Figures\n\n" + "\n".join(
        f"![{os.path.basename(f)}]({os.path.basename(f)})" for f in figs) + "\n")

    md = "\n".join(s)
    path = os.path.join(outdir, "interpretation.md")
    open(path, "w").write(md)
    if verbose:
        print(md)
    return {"markdown_path": path, "figures": figs, "markdown": md}
