import pandas as pd
import hmmlearn.hmm as hmm
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Define the main backtesting function

def run_backtest(data_df, feature_list, backtest_signal_start_date_str, num_lead):
    """Runs the event-driven backtest. Signals start from backtest_signal_start_date_str."""
    # Define backtesting parameters
    window_size = 4 * 252  # Rolling window for historical data (4 years)
    train_test_split_days = 90  # Days for the test set
    min_samples_for_rf = 30  # Minimum samples to train a Random Forest model
    
    # Initialize a 'signal' column in the DataFrame
    data_df['signal'] = 0.0

    # Determine the first possible index based on the historical window size
    first_possible_loop_idx_due_to_window = window_size
    target_signal_start_datetime = pd.to_datetime(backtest_signal_start_date_str)
    
    # Find all dates in the index that are on or after the target start date
    dates_after_target = data_df.index[data_df.index >= target_signal_start_datetime]

    # Handle case where start date is after all available data
    if dates_after_target.empty:
        print(f"Target signal start date {backtest_signal_start_date_str} is after the last data point. No backtest performed.")
        return data_df

    # Find the integer location of the first target date
    first_target_date_idx = data_df.index.get_loc(dates_after_target[0])

    # The main loop starts at the later of the two possible start dates
    loop_start_index = max(first_possible_loop_idx_due_to_window, first_target_date_idx)
    print(f"Backtest signal generation will start from: {data_df.index[loop_start_index].strftime('%Y-%m-%d')}")

    # Main backtesting loop
    for t in range(loop_start_index, len(data_df)):
        current_prediction_date = data_df.index[t]
        # Extract the historical data sample for this iteration
        data_sample = data_df.iloc[t - window_size : t].copy()

        if len(data_sample) < window_size:
            continue

        # Split the sample into training and test sets
        model_training_data = data_sample.iloc[:-train_test_split_days].copy()

        # Ensure there is enough data to train
        if len(model_training_data) < (min_samples_for_rf * 2):
            data_df.loc[current_prediction_date, 'signal'] = 0.0
            continue

        # --- HMM for Regime Detection ---
        hmm_model = None
        hmm_train_features = model_training_data[['returns']].copy()
        hmm_train_features.dropna(inplace=True)

        if len(hmm_train_features) >= min_samples_for_rf:
            hmm_model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=100, random_state=100)
            hmm_model.fit(hmm_train_features)
            regimes_for_rf_training = hmm_model.predict(hmm_train_features)
            model_training_data.loc[hmm_train_features.index, 'regime'] = regimes_for_rf_training

        # --- Regime-Specific Random Forest Models ---
        model0, model1 = None, None
        if 'regime' in model_training_data.columns and feature_list:
            regime_0_data = model_training_data[model_training_data['regime'] == 0].copy()
            regime_1_data = model_training_data[model_training_data['regime'] == 1].copy()

            # Train Model for Regime 0
            if len(regime_0_data) >= min_samples_for_rf and len(regime_0_data['y_signal'].unique()) > 1:
                X0, y0 = regime_0_data[feature_list].iloc[:-num_lead,:], regime_0_data['y_signal'].iloc[:-num_lead]
                model0 = RandomForestClassifier(n_estimators=50, random_state=100)
                model0.fit(X0, y0)

            # Train Model for Regime 1
            if len(regime_1_data) >= min_samples_for_rf and len(regime_1_data['y_signal'].unique()) > 1:
                X1, y1 = regime_1_data[feature_list].iloc[:-num_lead,:], regime_1_data['y_signal'].iloc[:-num_lead]
                model1 = RandomForestClassifier(n_estimators=50, random_state=100)
                model1.fit(X1, y1)

        
        # --- Prediction for the current day ---
        features_for_pred_day = data_sample[feature_list].iloc[-num_lead:].copy() if feature_list else pd.DataFrame()
        hmm_features_for_pred_day = data_sample[['returns']].iloc[-1:].copy()
        next_day_regime_probs = np.array([0.5, 0.5]) # Default probabilities

        # Predict next day's regime probability using HMM transition matrix
        if hmm_model and not hmm_features_for_pred_day.isnull().values.any() and hasattr(hmm_model, 'transmat_'):
            last_day_state_probs = hmm_model.predict_proba(hmm_features_for_pred_day)[0]
            next_day_regime_probs = last_day_state_probs @ hmm_model.transmat_

        # Generate signals from each model
        signal0, signal1 = 0.0, 0.0
        
        # Check if model0 can predict
        if model0 and not features_for_pred_day.empty and \
           not (features_for_pred_day.isnull().values.any() or np.isinf(features_for_pred_day.values).any()):
            signal0 = model0.predict_proba(features_for_pred_day)[0][1]
        
        # Check if model1 can predict
        if model1 and not features_for_pred_day.empty and \
           not (features_for_pred_day.isnull().values.any() or np.isinf(features_for_pred_day.values).any()):
            signal1 = model1.predict_proba(features_for_pred_day)[0][1]

        # Combine signals based on regime probability
        if next_day_regime_probs[0] > next_day_regime_probs[1]:
            final_signal = signal0
        elif next_day_regime_probs[1] > next_day_regime_probs[0]:
            final_signal = signal1
        else: # If probabilities are equal, remain neutral
            final_signal = 0.0
       # Limit threshold for the signal probability as a risk management tool
        limit = 0.03 # 3 is best up to now
        # Store the final signal in the DataFrame
        data_df.loc[current_prediction_date, 'signal'] = 1 if final_signal>(0.5+limit) else -1 if final_signal<(0.5-limit) else 0

        if t % 100 == 0:
            print(f"Processed up to {current_prediction_date.strftime('%Y-%m-%d')}, Signal: {final_signal}")

    return data_df