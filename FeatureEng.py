# Define a function to engineer technical features and ensure stationarity
import pandas as pd
import numpy as np
from ta import add_all_ta_features
from statsmodels.tsa.stattools import adfuller

def engineer_features(data_df, num_lead):
    """Engineers technical features and ensures stationarity."""
    # Note: The 'returns' column is expected to be present from get_data function.

    # Create a new DataFrame with the same index as the input DataFrame for TA feature calculation
    df_input_for_ta = pd.DataFrame(index=data_df.index)
    # Copy 'Open', 'High', 'Low', 'Close', and 'Volume' prices, casting to float
    df_input_for_ta['Open'] = data_df['Open'].astype(float)
    df_input_for_ta['High'] = data_df['High'].astype(float)
    df_input_for_ta['Low'] = data_df['Low'].astype(float)
    df_input_for_ta['Close'] = data_df['Close'].astype(float)
    df_input_for_ta['Volume'] = data_df['Volume'].astype(float)

    # Add all technical analysis features using the 'ta' library
    features_added_df = add_all_ta_features(
        df_input_for_ta,
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
        fillna=True
    )

    # Define a list of original OHLCV columns to remove from the TA features DataFrame
    columns_to_drop_from_ta_output = ["Open", "High", "Low", "Close", "Volume"]
    # Drop the original OHLCV columns to keep only the TA features
    processed_features_df = features_added_df.drop(columns=columns_to_drop_from_ta_output, errors='ignore')

    # Initialize lists to categorize indicators
    indicators_to_become_stationary = []
    indicators_to_drop = []

    # Loop through each indicator to check for stationarity
    for indicator in processed_features_df.columns:
        indicator_series = processed_features_df[indicator].replace([np.inf, -np.inf], np.nan).dropna()
        
        # Drop indicators with insufficient data for the ADF test
        if len(indicator_series) < 20:
            indicators_to_drop.append(indicator)
            continue
            
        # Perform the Augmented Dickey-Fuller test
        pvalue = adfuller(indicator_series, regression='c', autolag='AIC')[1]
        
        # Check if the series is non-stationary
        if pvalue > 0.05:
            indicators_to_become_stationary.append(indicator)
        # Drop indicators where the p-value calculation fails
        elif np.isnan(pvalue):
            indicators_to_drop.append(indicator)

    # Make non-stationary indicators stationary using percentage change
    if indicators_to_become_stationary:
        processed_features_df.loc[:, indicators_to_become_stationary] = processed_features_df[indicators_to_become_stationary].pct_change()

    # Drop indicators marked for removal
    if indicators_to_drop:
        processed_features_df.drop(columns=indicators_to_drop, inplace=True, errors='ignore')

    # Concatenate the original data with the new features
    data_with_all_features = pd.concat([data_df, processed_features_df], axis=1)

    # Define the prediction target 'y_signal'
    data_with_all_features['y_signal'] = np.where(data_with_all_features['returns'].shift(-num_lead)>0,1,0)
    
    # Identify the final list of feature columns
    final_feature_columns = [col for col in processed_features_df.columns if col in data_with_all_features.columns]
    return data_with_all_features, final_feature_columns