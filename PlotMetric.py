import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# Define a function to plot the backtesting results
def plot_results(results_df, ticker):
    """Plots cumulative returns."""
    # Check if the results DataFrame is empty
    if results_df.empty:
        print(f"No results to plot for {ticker}.")
        return

    # Calculate cumulative returns for a Buy-and-Hold strategy
    results_df['bh_cum_rets'] = (1 + results_df['returns']).cumprod()
    # Calculate strategy returns by multiplying daily returns with the trading signal
    results_df['strategy_returns'] = results_df['returns'] * results_df['signal']
    results_df.dropna(subset=['strategy_returns'], inplace=True)

    # Check if there are any valid returns left to plot
    if results_df.empty or results_df['strategy_returns'].isnull().all():
        print(f"No valid strategy returns to plot for {ticker} after processing.")
        return

    # Calculate cumulative returns for the HMM-RF strategy
    results_df['strategy_cum_rets'] = (1 + results_df['strategy_returns']).cumprod()

    # Create the plot
    plt.figure(figsize=(15, 7))
    plt.plot(results_df['bh_cum_rets'], label="Buy-and-Hold")
    plt.plot(results_df['strategy_cum_rets'], label="HMM-RF Strategy")
    plt.title(f'{ticker} Strategy Cumulative Returns', fontsize=16)
    plt.xlabel('Date', fontsize=15)
    plt.ylabel('Cumulative Returns', fontsize=15)
    plt.tick_params(axis='both', labelsize=15)
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()