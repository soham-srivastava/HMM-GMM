import pandas as pd 
import numpy as np
import yfinance as yf

def get_data(ticker, start_date, end_date): # yf data not pandas 
    """Downloads and prepares stock data."""
    # Download historical stock data using yfinance
    data = yf.download(ticker, start=start_date, end=end_date, group_by='ticker')[ticker]
    # Calculate daily percentage returns and store them in a new 'returns' column
    data['returns'] = data['Close'].pct_change()
    # Return the DataFrame with historical data and returns
    return data