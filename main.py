import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from ta import add_all_ta_features
from sklearn.ensemble import RandomForestClassifier
import datetime
# Import the hmm module from hmmlearn for Hidden Markov Models
from hmmlearn import hmm
from statsmodels.tsa.stattools import adfuller

# Install pyfolio-reloaded to check strategy performance metrics
import pyfolio.timeseries as pf_ts



from Strategy import run_backtest
from FeatureEng import engineer_features
from PlotMetric import plot_results
from Metric_display import compute_perf_stats
from Fetch_data import get_data


# Main execution block: This code runs when the script is executed directly
if __name__ == '__main__':
    # Define script parameters
    TICKER = 'BTC-USD'
    START_DATE = '2008-01-01'
    END_DATE = '2025-06-18'
    BACKTEST_SIGNAL_START_DATE = '2024-01-01'
    NUM_LEAD = 1

    # 1. Download and prepare the raw historical data
    raw_data = get_data(TICKER, START_DATE, END_DATE)
    
    # 2. Engineer features from the raw data
    data_with_features, feature_cols = engineer_features(raw_data.copy(), NUM_LEAD)

    print(f"Data prepared. Number of features: {len(feature_cols)}")
    print(f"Data shape after preprocessing: {data_with_features.shape}")

    # 3. Run the backtest
    results_df = run_backtest(data_with_features.copy(), feature_cols, BACKTEST_SIGNAL_START_DATE, NUM_LEAD)
    
    # 4. Filter results for the plotting period
    results_to_plot = results_df[results_df.index >= pd.to_datetime(BACKTEST_SIGNAL_START_DATE)].copy()

    # 5. Plot the results
    plot_results(results_to_plot, TICKER)