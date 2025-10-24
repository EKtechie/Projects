# train.py
# Description: Trains the XGBoost model, evaluates it, and saves the artifact.

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import config
import feature_engineering

def split_data(df):
    """Splits the data into training and testing sets based on a date."""
    
    # Define target and features
    X = df.drop(config.TARGET_COL, axis=1)
    Y = df[config.TARGET_COL]
    
    # Split based on date
    X_train = X.loc[:config.SPLIT_DATE]
    Y_train = Y.loc[:config.SPLIT_DATE]
    
    X_test = X.loc[config.TEST_START_DATE:]
    Y_test = Y.loc[config.TEST_START_DATE:]
    
    print(f"Training set shape: {X_train.shape}, {Y_train.shape}")
    print(f"Test set shape: {X_test.shape}, {Y_test.shape}")
    
    return X_train, Y_train, X_test, Y_test

def train_model(X_train, Y_train, X_test, Y_test):
    """Initializes and trains the XGBoost model."""
    
    print("Training XGBoost model...")
    xgb = XGBRegressor(**config.XGB_PARAMS)
    
    xgb.fit(X_train, Y_train,
            eval_set=[(X_train, Y_train), (X_test, Y_test)],
            verbose=False)
    
    print("Model training complete.")
    return xgb

def evaluate_model(model, X_test, Y_test):
    """Evaluates the model on the test set and prints metrics."""
    
    predictions = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(Y_test, predictions))
    mae = mean_absolute_error(Y_test, predictions)
    
    print(f"--- Model Evaluation ---")
    print(f"XGBoost RMSE: {rmse}")
    print(f"XGBoost MAE : {mae}")
    
    return predictions

def main():
    """Main training pipeline execution."""
    
    # 1. Get processed data
    data = feature_engineering.get_processed_data(config.DATA_PATH)
    
    # 2. Split data
    X_train, Y_train, X_test, Y_test = split_data(data)
    
    # 3. Train model
    model = train_model(X_train, Y_train, X_test, Y_test)
    
    # 4. Evaluate model
    evaluate_model(model, X_test, Y_test)
    
    # 5. Save model
    joblib.dump(model, config.MODEL_PATH)
    print(f"Model saved to {config.MODEL_PATH}")

if __name__ == "__main__":
    main()