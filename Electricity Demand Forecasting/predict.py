# predict.py
# Description: Loads the trained model and makes predictions on new data.

import pandas as pd
import joblib
import config
import feature_engineering

def load_model(path):
    """Loads a saved model artifact."""
    model = joblib.load(path)
    return model

def make_predictions(data, model):
    """Makes predictions on a set of features."""
    
    # Select the test set features (same logic as in train.py)
    try:
        X_test = data.loc[config.TEST_START_DATE:].drop(config.TARGET_COL, axis=1)
        Y_test_actual = data.loc[config.TEST_START_DATE:][config.TARGET_COL]
    except KeyError:
        print("Test data range not found. Predicting on all available data after split.")
        X_test = data.loc[config.TEST_START_DATE:]
        # Handle case where target column might not be present in new data
        if config.TARGET_COL in X_test.columns:
            X_test = X_test.drop(config.TARGET_COL, axis=1)
        Y_test_actual = None
        
    if X_test.empty:
        print("No data found after the split date for prediction.")
        return
        
    print(f"Making predictions on {X_test.shape[0]} samples...")
    predictions = model.predict(X_test)
    
    # Create a DataFrame with predictions
    results = pd.DataFrame(index=X_test.index)
    if Y_test_actual is not None:
        results['Actual_Demand'] = Y_test_actual
    results['Predicted_Demand'] = predictions
    
    return results

def main():
    """Main prediction pipeline."""
    
    # 1. Load model
    try:
        model = load_model(config.MODEL_PATH)
    except FileNotFoundError:
        print(f"Error: Model file not found at {config.MODEL_PATH}")
        print("Please run train.py to create the model file.")
        return

    # 2. Get processed data
    # Note: This is crucial. We process *all* data to ensure lags/rolling features
    # are calculated correctly, even for the first rows of the test set.
    data = feature_engineering.get_processed_data(config.DATA_PATH)

    # 3. Make predictions
    results = make_predictions(data, model)
    
    # 4. Save results
    if results is not None:
        results.to_csv(config.PREDICTIONS_PATH)
        print(f"Predictions saved to {config.PREDICTIONS_PATH}")
        print(results.head())

if __name__ == "__main__":
    main()