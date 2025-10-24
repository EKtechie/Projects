# feature_engineering.py
# Description: Loads raw data, cleans it, and creates time-series features.

import pandas as pd
import numpy as np
import config

def load_data(path):
    """Loads raw data from a CSV file."""
    data = pd.read_csv(path)
    return data

def preprocess(data):
    """Handles data type conversion, indexing, and missing value imputation."""
    # Convert Timestamp to datetime and set as index
    data[config.TIMESTAMP_COL] = pd.to_datetime(data[config.TIMESTAMP_COL])
    data = data.set_index(config.TIMESTAMP_COL)
    
    # Drop rows where all values are missing
    data = data.dropna(how='all')
    
    # Impute missing values
    data[config.FFILL_COLS] = data[config.FFILL_COLS].ffill()
    data[config.BFILL_COLS] = data[config.BFILL_COLS].bfill()
    data[config.INTERPOLATE_COL] = data[config.INTERPOLATE_COL].interpolate(method='time')
    
    return data

def create_time_series_features(data):
    """Creates new features based on the datetime index."""
    df = data.copy()
    
    # Create time-based features
    df['quarter'] = df.index.quarter
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
    
    # Convert relevant columns to integer
    int_cols = ['hour', 'dayofweek', 'month', 'year', 'dayofyear']
    df[int_cols] = df[int_cols].astype(int)
    
    return df

def create_lag_rolling_features(data):
    """Creates lag and rolling window features for the target variable."""
    df = data.copy()
    
    # Create lag features
    for col_name, lag in config.LAG_FEATURES.items():
        df[col_name] = df[config.TARGET_COL].shift(lag)
        
    # Create rolling features
    for col_name, window in config.ROLLING_FEATURES.items():
        if 'mean' in col_name:
            df[col_name] = df[config.TARGET_COL].rolling(window=window).mean()
        if 'std' in col_name:
            df[col_name] = df[config.TARGET_COL].rolling(window=window).std()
            
    # Drop NaNs created by lag/rolling features
    df = df.dropna()
    return df

def get_processed_data(path):
    """Main function to run the full data processing pipeline."""
    raw_data = load_data(path)
    data_preprocessed = preprocess(raw_data)
    data_time_features = create_time_series_features(data_preprocessed)
    data_final_features = create_lag_rolling_features(data_time_features)
    return data_final_features

if __name__ == "__main__":
    # This block runs if the script is executed directly
    print("Starting data processing...")
    
    # Process the data
    processed_data = get_processed_data(config.DATA_PATH)
    
    # Save the processed data to an Excel file (as in the notebook)
    # Note: Saving to Excel resets the index, so we reset it here for the export.
    processed_data.reset_index().to_excel(config.PROCESSED_DATA_PATH, index=False)
    
    print(f"Data processed and saved to {config.PROCESSED_DATA_PATH}")
    print(processed_data.info())