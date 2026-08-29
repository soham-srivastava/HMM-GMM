"""
HMM-gated, regime-conditional Random Forest strategy.

    Fetch_data      OHLCV
    FeatureEng      causal features (transform plan frozen in-sample)
    RegimeModel     Gaussian HMM gate, canonical states, correct filtering
    Strategy        walk-forward: gate -> experts -> blended probability
    Metrics         turnover-aware pnl and classification stats
    Interpretation  what the model learned and whether to believe it

Run:  python main.py
"""

import pandas as pd

import config as C
from Fetch_data import get_data
from FeatureEng import engineer_features
from Strategy import run_backtest
from Metrics import (attach_realised, summary_table, trade_stats, apply_costs,
                     classification_stats, cost_sweep, breakeven_cost_bps,
                     delay_sensitivity)
from Interpretation import generate_report


def main():
    assert pd.to_datetime(C.STATIONARITY_FIT_END) <= pd.to_datetime(C.SIGNAL_START), \
        "the transform plan must be fitted before the evaluation window"

    raw = get_data(C.TICKER, C.START_DATE, C.END_DATE)
    print(f"[main] {C.TICKER}: {len(raw)} bars "
          f"{raw.index.min().date()} -> {raw.index.max().date()}")

    data, feats, hmm_cols, fa = engineer_features(
        raw.copy(), C.NUM_LEAD, fit_end=C.STATIONARITY_FIT_END)
    print(f"[main] {len(feats)} usable features | HMM observes {hmm_cols}")

    results, art = run_backtest(data.copy(), feats, hmm_cols, C.SIGNAL_START,
                                num_lead=C.NUM_LEAD)

    diag = art.get("diagnostics", pd.DataFrame())
    ev = attach_realised(results.loc[results.index >= diag.index.min()].copy())

    print("\n=== PERFORMANCE ===")
    print(summary_table(ev, C.COST_BPS).to_string())
    print("\n=== TRADING ===")
    print(pd.Series(trade_stats(apply_costs(ev, C.COST_BPS))).round(3).to_string())
    print("\n=== CLASSIFICATION ===")
    print(pd.Series(classification_stats(ev)).round(4).to_string())
    print("\n=== COST SENSITIVITY ===")
    print(cost_sweep(ev).round(3).to_string())
    print(f"Break-even: {breakeven_cost_bps(ev):.1f} bps per unit turnover")
    print("\n=== DELAY SENSITIVITY (leakage smoke test) ===")
    print(delay_sensitivity(ev, bps=C.COST_BPS).round(3).to_string())

    generate_report(ev, art, fa["report"], ticker=C.TICKER, bps=C.COST_BPS)
    return ev, art


if __name__ == "__main__":
    main()
