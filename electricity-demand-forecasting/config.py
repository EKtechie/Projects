# config.py
# Description: Configuration file for the electricity demand forecasting pipeline.

# config.py

# File Paths 
# Use relative paths from the root of the project folder
DATA_PATH = "data/Electricity+Demand+Dataset.csv"
MODEL_PATH = "models/electricity_xgb_prediction_model.pkl"
PROCESSED_DATA_PATH = "outputs/processed_electricity_demand.xlsx"
PREDICTIONS_PATH = "outputs/predictions.csv"

# Data Processing 
TIMESTAMP_COL = 'Timestamp'
TARGET_COL = 'Demand'

# Columns for missing value imputation
FFILL_COLS = ['hour', 'dayofweek', 'month', 'year', 'dayofyear']
BFILL_COLS = ['Temperature', 'Humidity']
INTERPOLATE_COL = 'Demand'

# --- Feature Engineering ---
LAG_FEATURES = {
    'Demand_lag_24hr': 24,
    'demand_lag_168hr': 168
}

ROLLING_FEATURES = {
    'demand_rolling_mean_24hr': 24,
    'demand_rolling_std_24hr': 24
}

# --- Model Training ---
SPLIT_DATE = '2023-12-31'
TEST_START_DATE = '2024-01-01'

XGB_PARAMS = {
    'n_estimators': 1000,
    'early_stopping_rounds': 50,
    'learning_rate': 0.02,
    'random_state': 40,
    'objective': 'reg:squarederror'
}