"""
Checks on synthetic data.  Run: python test_pipeline.py

TEST 4 is the important one: rebuilding features and re-running the whole
walk-forward on data truncated 60 bars early must reproduce the earlier
signals exactly.  If any future information reaches a past decision the two
runs diverge and the assert fires.
"""

import numpy as np
import pandas as pd
import scipy.stats as st

from FeatureEng import engineer_features
from RegimeModel import fit_regime_model
from Strategy import run_backtest
from Metrics import attach_realised, summary_table

rng = np.random.default_rng(7)
FIT_END, SIG_START, WIN, N = "2019-01-01", "2022-01-01", 500, 3


def make_data(n=2000):
    """Three regimes: bear/volatile, choppy/flat, bull/calm."""
    A = np.array([[.95, .04, .01], [.03, .94, .03], [.01, .04, .95]])
    mu, sd = np.array([-.0025, .0000, .0022]), np.array([.030, .018, .011])
    s, states, rets = 0, [], []
    for _ in range(n):
        s = rng.choice(3, p=A[s]); states.append(s)
        rets.append(mu[s] + sd[s] * rng.standard_normal())
    rets = np.array(rets); close = 100 * np.cumprod(1 + rets)
    idx = pd.bdate_range("2016-01-04", periods=n)
    op = np.r_[close[0], close[:-1]]
    df = pd.DataFrame({
        "Open": op,
        "High": np.maximum(close * (1 + abs(rng.normal(0, .004, n))), np.maximum(op, close)),
        "Low": np.minimum(close * (1 - abs(rng.normal(0, .004, n))), np.minimum(op, close)),
        "Close": close, "Volume": rng.lognormal(12, .4, n)}, index=idx)
    df.index.name = "Date"; df["returns"] = df["Close"].pct_change()
    return df, pd.Series(states, index=idx)


raw, truth = make_data()
data, feats, hmm_cols, fa = engineer_features(raw.copy(), 1, fit_end=FIT_END)

# [1] label alignment
ok = (data["y_signal"].iloc[:-1].to_numpy() ==
      (data["returns"].shift(-1).iloc[:-1] > 0).astype(float).to_numpy())
assert ok.all() and data["y_signal"].isna().sum() == 1
print(f"[1] {len(feats)} features, hmm={hmm_cols}, label alignment OK")

# [2] HMM: canonical order + filtering matches a manual forward recursion
X = data[hmm_cols].replace([np.inf, -np.inf], np.nan).dropna().iloc[:900]
g = fit_regime_model(X, N, 0)
assert np.all(np.diff(g.means[:, 0]) >= 0), "states not sorted by mean"
lg = np.column_stack([st.multivariate_normal(g.means[k], np.diag(g.variances[k]),
                                             allow_singular=True).logpdf(X.to_numpy())
                      for k in range(N)])
a = np.exp(lg[0] - lg[0].max()); a /= a.sum()
for i in range(1, len(X)):
    a = (a @ g.transmat) * np.exp(lg[i] - lg[i].max()); a /= a.sum()
assert np.allclose(a, g.posteriors(X)[-1], atol=1e-5), "filtering mismatch"
print(f"[2] filtering OK | {g.persistence()} \n{g.summary(252).round(4).to_string()}")

# [3] backtest runs
res, art = run_backtest(data.copy(), feats, hmm_cols, SIG_START, 1,
                        window_size=WIN, n_states=N, refit_every=5, verbose=False)
diag = art["diagnostics"]
assert len(diag) > 100
ev = attach_realised(res[res.index >= diag.index.min()].copy())
print(f"[3] {len(diag)} decisions\n{summary_table(ev, 10.0, 252).to_string()}")

# [4] causality
cut = data.index[-60]
d2, f2, h2, _ = engineer_features(raw.loc[:cut].copy(), 1, plan=fa["plan"])
_, art2 = run_backtest(d2.copy(), f2, h2, SIG_START, 1, window_size=WIN,
                       n_states=N, refit_every=5, verbose=False)
c = art2["diagnostics"].index.intersection(diag.index)
bad = int((~np.isclose(diag.loc[c, "prob_up"], art2["diagnostics"].loc[c, "prob_up"],
                       atol=1e-12)).sum())
print(f"[4] CAUSALITY: {len(c)} overlapping decisions, {bad} mismatches")
assert bad == 0, "LOOK-AHEAD: truncating the future changed past signals"

print("\nALL TESTS PASSED")
