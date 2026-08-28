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
def get_data_from_csv(file_path):
    """Reads stock data from a CSV file."""
    # Read the CSV file into a DataFrame, parsing the 'Date' column as datetime and setting it as the index
    data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    # Calculate daily percentage returns and store them in a new 'returns' column
    data['returns'] = data['Close'].pct_change()
    # Return the DataFrame with historical data and returns
    return data