import pyfolio.timeseries as pf_ts
import pandas as pd
import numpy as np

def compute_perf_stats(results_to_plot):
    # --- Calculate Performance Statistics ---
    strategy_stats = pf_ts.perf_stats(results_to_plot['strategy_returns'])
    bh_stats = pf_ts.perf_stats(results_to_plot['bh_cum_rets'].pct_change().fillna(0))
    
    # --- Combine Statistics into a Single Table ---
    # Concatenate the two series into a single DataFrame.
    perf_table = pd.concat([bh_stats, strategy_stats], axis=1)
    perf_table.columns = ['Buy & Hold', 'Strategy'] # Set the column names
    
    # --- Filter for Desired Metrics ---
    # Create a list of the specific metrics you want to see.
    desired_metrics = [
        'Annual return',
        'Cumulative returns',
        'Annual volatility',
        'Sharpe ratio',
        'Calmar ratio',
        'Max drawdown',
        'Sortino ratio',
    ]
    
    # Select only the rows that match your desired metrics.
    final_table = perf_table.loc[desired_metrics]
    
    for index in final_table.index:
        if index in ['Annual return', 'Cumulative returns', 'Annual volatility', 'Max drawdown']:
            for column in final_table.columns:
                final_table.loc[index,column] = f'{round(final_table.loc[index,column]*100,2):.2f}%'
        else:
            for column in final_table.columns:
                final_table.loc[index,column] = f'{round(final_table.loc[index,column],2):.2f}'
    
    # Print the table
    print(final_table)