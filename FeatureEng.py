"""
Causal feature engineering.

Two rules:
  1. The transform plan is fitted ONCE on data before STATIONARITY_FIT_END and
     frozen.  Applying it (pct_change / diff) only looks backwards, so no
     future information reaches a past feature.
  2. pct_change does not guarantee stationarity, so every series is re-tested
     after transformation.  ADF decides; KPSS is reported next to it.
"""

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from ta import add_all_ta_features

from config import ADF_PVALUE, DROP_NONSTATIONARY, VOL_WINDOW, HMM_FEATURES

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def add_ta_block(df):
    """The ~85 `ta` indicators, with the raw OHLCV columns stripped back out."""
    px = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce").astype(float)
                       for c in OHLCV}, index=df.index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feats = add_all_ta_features(px, open="Open", high="High", low="Low",
                                    close="Close", volume="Volume", fillna=True)
    return feats.drop(columns=OHLCV, errors="ignore")


def _test(s):
    """(ADF p, KPSS p). ADF null = unit root; KPSS null = stationary."""
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 40 or s.nunique() < 3:
        return np.nan, np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            a = adfuller(s, regression="ct", autolag="AIC")[1]
        except Exception:
            a = np.nan
        try:
            k = kpss(s, regression="c", nlags="auto")[1]
        except Exception:
            k = np.nan
    return a, k


def fit_transform_plan(raw, fit_index):
    """Decide each feature's transform using only rows in `fit_index`."""
    plan, rows = {}, []
    fit = raw.loc[raw.index.isin(fit_index)]

    for col in raw.columns:
        s = fit[col]
        if s.dropna().nunique() <= 2:                    # 0/1 flags are stationary
            plan[col] = "raw"
            rows.append((col, "raw", np.nan, np.nan, "binary flag"))
            continue

        adf0, kpss0 = _test(s)
        if np.isnan(adf0):
            plan[col] = "drop"
            rows.append((col, "drop", adf0, kpss0, "untestable"))
            continue

        if adf0 < ADF_PVALUE:
            plan[col] = "raw"
            rows.append((col, "raw", adf0, kpss0, "stationary as-is"))
            continue

        # non-stationary: pct_change if strictly positive, else first difference
        kind = "pct_change" if (s.dropna() > 0).all() else "diff"
        t = (s.pct_change() if kind == "pct_change" else s.diff())
        adf1, _ = _test(t.replace([np.inf, -np.inf], np.nan))
        if not np.isnan(adf1) and adf1 < ADF_PVALUE:
            plan[col] = kind
            rows.append((col, kind, adf0, adf1, "stationary after transform"))
        elif DROP_NONSTATIONARY:
            plan[col] = "drop"
            rows.append((col, "drop", adf0, adf1, "still non-stationary"))
        else:
            plan[col] = kind
            rows.append((col, kind, adf0, adf1, "kept (weak)"))

    report = pd.DataFrame(rows, columns=["feature", "transform", "adf_p_before",
                                         "adf_p_after", "status"]).set_index("feature")
    return plan, report


def apply_transform_plan(raw, plan):
    out = {}
    for col, kind in plan.items():
        if kind == "drop" or col not in raw:
            continue
        s = raw[col]
        if kind == "pct_change":
            s = s.pct_change()
        elif kind == "diff":
            s = s.diff()
        out[col] = s.replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame(out, index=raw.index)


def build_hmm_features(df):
    """
    What the regime model observes.  Deliberately small: an HMM is a density
    model, and 85 correlated indicators would make it unidentifiable.
    """
    cols = {"returns": df["returns"]}
    if "vol" in HMM_FEATURES:
        cols["vol"] = df["returns"].rolling(VOL_WINDOW).std()
    return pd.DataFrame({k: cols[k] for k in HMM_FEATURES}, index=df.index)


def engineer_features(df, num_lead, fit_end=None, plan=None):
    """
    Returns (data, feature_cols, hmm_cols, report).

    Label: y_t = 1[ Close_{t+num_lead}/Close_t - 1 > 0 ].
    """
    raw = add_ta_block(df)
    if plan is None:
        idx = raw.index[raw.index < pd.to_datetime(fit_end)]
        plan, report = fit_transform_plan(raw, idx)
    else:
        report = pd.DataFrame()

    features = apply_transform_plan(raw, plan)
    hmm = build_hmm_features(df)
    out = pd.concat([df, features, hmm.drop(columns=["returns"], errors="ignore")], axis=1)

    fwd = out["Close"].shift(-num_lead) / out["Close"] - 1.0
    out["y_signal"] = np.where(fwd > 0, 1.0, 0.0)
    out.loc[fwd.isna(), "y_signal"] = np.nan          # unknowable labels stay NaN

    return out, list(features.columns), list(hmm.columns), {"plan": plan, "report": report}
