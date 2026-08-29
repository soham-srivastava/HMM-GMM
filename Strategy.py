"""
Walk-forward backtest: HMM gate + one Random Forest expert per regime.

For each bar t (information set = everything up to t-1):

    window   = df.iloc[t-W : t]                 rows t-W .. t-1
    gate     : fit HMM on window observables, filter, P(S_t) = P(S_{t-1}) @ A
    experts  : one RF per regime, trained on window.iloc[:-num_lead]
               (the last num_lead rows have no known label yet)
    signal   : P(up) = sum_s P(S_t = s) * p_s, then a dead band

Why the probability-weighted blend rather than picking the argmax expert:
argmax at P = (0.34, 0.33, 0.33) manufactures conviction the gate does not
have.  On BTC the blend measured Sharpe 1.02 vs 0.61 for argmax, with lower
turnover.  An expert that could not be trained returns 0.5 -- no opinion --
never 0.0, which would read as a maximum-conviction short.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from RegimeModel import fit_regime_model
from config import (WINDOW_SIZE, REFIT_EVERY, MIN_SAMPLES_PER_REGIME, N_STATES,
                    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLES_LEAF,
                    RANDOM_STATE, SIGNAL_THRESHOLD)


def _fit_expert(X, y):
    if len(X) < MIN_SAMPLES_PER_REGIME or y.nunique() < 2:
        return None
    rf = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
                                min_samples_leaf=RF_MIN_SAMPLES_LEAF,
                                random_state=RANDOM_STATE, n_jobs=-1)
    return rf.fit(X, y)


def _prob_up(model, x):
    """P(y=1). 0.5 means 'no opinion' -- what a missing expert must say."""
    if model is None or not np.isfinite(x.to_numpy(dtype=float)).all():
        return 0.5
    classes = list(model.classes_)
    if 1.0 not in classes:
        return 0.5
    return float(model.predict_proba(x)[0][classes.index(1.0)])


def _entropy(p):
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def run_backtest(df, feature_cols, hmm_cols, signal_start, num_lead=1,
                 window_size=WINDOW_SIZE, n_states=N_STATES,
                 refit_every=REFIT_EVERY, threshold=SIGNAL_THRESHOLD, verbose=True):
    """Returns (results_df, artifacts)."""
    df = df.copy()
    df["signal"] = np.nan
    df["prob_up"] = np.nan

    after = df.index[df.index >= pd.to_datetime(signal_start)]
    if after.empty:
        return df, {}
    start = max(window_size, df.index.get_loc(after[0]))
    if verbose:
        print(f"[Strategy] {df.index[start].date()} -> {df.index[-1].date()}  "
              f"| states={n_states} window={window_size} refit/{refit_every}")

    diag, imps, gates = [], [], []
    gate, experts, n_train = None, [None] * n_states, [np.nan] * n_states
    since_refit = np.inf

    for t in range(start, len(df)):
        date = df.index[t]
        window = df.iloc[t - window_size:t]                      # info <= t-1
        obs = window[hmm_cols].replace([np.inf, -np.inf], np.nan).dropna()
        if len(obs) < 100:
            df.loc[date, ["signal", "prob_up"]] = [0.0, 0.5]
            continue

        refit = since_refit >= refit_every or gate is None
        if refit:
            gate = fit_regime_model(obs, n_states, RANDOM_STATE)
        if gate is None:
            df.loc[date, ["signal", "prob_up"]] = [0.0, 0.5]
            since_refit = np.inf
            continue

        post = gate.posteriors(obs)
        p_state = post[-1] @ gate.transmat                       # P(S_t | info<=t-1)

        if refit:
            train = window.iloc[:-num_lead] if num_lead else window
            train = train[feature_cols + ["y_signal"]].replace(
                [np.inf, -np.inf], np.nan).dropna()
            pt = pd.DataFrame(post, index=obs.index).reindex(train.index).dropna()
            train = train.loc[pt.index]
            state_of_row = pt.to_numpy().argmax(1)

            experts, n_train = [], []
            for s in range(n_states):
                m = state_of_row == s
                experts.append(_fit_expert(train.loc[m, feature_cols],
                                           train.loc[m, "y_signal"]))
                n_train.append(int(m.sum()))
                if experts[s] is not None:
                    imps.append(pd.Series(experts[s].feature_importances_,
                                          index=feature_cols, name=(date, s)))
            gates.append(gate.summary().assign(date=date, **gate.persistence()))
        since_refit = 1 if refit else since_refit + 1

        x = window[feature_cols].iloc[[-1]].replace([np.inf, -np.inf], np.nan)
        p_exp = [_prob_up(m, x) for m in experts]
        prob_up = float(np.dot(p_state, p_exp))
        signal = 1.0 if prob_up > 0.5 + threshold else -1.0 if prob_up < 0.5 - threshold else 0.0
        df.loc[date, ["signal", "prob_up"]] = [signal, prob_up]

        row = {"date": date, "prob_up": prob_up, "signal": signal,
               "gate_entropy": _entropy(p_state)}
        for s in range(n_states):
            row[f"p_state_{s}"] = p_state[s]
            row[f"expert_{s}_prob_up"] = p_exp[s]
            row[f"expert_{s}_trained"] = experts[s] is not None
            row[f"n_train_{s}"] = n_train[s] if refit else np.nan
        diag.append(row)

        if verbose and (t - start) % 100 == 0:
            print(f"  {date.date()}  P(S)={np.round(p_state, 2)}  "
                  f"experts={np.round(p_exp, 2)}  P(up)={prob_up:.3f}  {signal:+.0f}")

    diagnostics = pd.DataFrame(diag).set_index("date") if diag else pd.DataFrame()
    importances = pd.DataFrame(imps) if imps else pd.DataFrame()
    if not importances.empty:
        importances.index = pd.MultiIndex.from_tuples(importances.index,
                                                      names=["date", "state"])
    return df, {"diagnostics": diagnostics, "importances": importances,
                "gate_history": pd.concat(gates) if gates else pd.DataFrame(),
                "n_states": n_states}
