Electricity Demand Forecasting

This project builds a complete machine learning pipeline to forecast electricity demand. It uses an XGBoost Regressor model trained on time-series data, including temporal features (hour, day, month) and weather data (temperature, humidity).

Features
Modular Pipeline: The project is split into separate, reusable Python scripts for clarity and maintainability.

Data Processing: Loads raw data, cleans missing values using various imputation techniques (ffill, bfill, interpolation), and sets a proper datetime index.

Feature Engineering: Creates a rich set of features crucial for time-series forecasting:

Temporal: Hour, day of week, month, year, day of year, quarter, and weekend flags.

Lag: Demand from 24 hours ago (daily seasonality) and 168 hours ago (weekly seasonality).

Rolling Window: 24-hour rolling mean and standard deviation to capture recent trends.

Model Training: Trains an XGBoost Regressor with early stopping to prevent overfitting.

Evaluation: Evaluates the model on a hold-out test set (all data from 2024) using Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).

Inference: Includes a separate script to load the saved model and generate predictions.

How to Run

1. Setup
Clone the repository: View Repository Here

Bash

git clone https://github.com/EKtechie/Projects.git
cd "Projects/Electricity Demand Forecasting"

Install dependencies:

Bash

pip install -r requirements.txt
Add Data: Place your Electricity+Demand+Dataset.csv file inside the data/ folder.

2. Run the Full Pipeline
You can now run the main scripts from your terminal.

Step 1: Train the Model This script will:

Load the raw data from data/.

Perform all feature engineering.

Train the XGBoost model.

Save the model artifact to models/electricity_xgb_prediction_model.pkl.

Save the processed data to outputs/processed_electricity_demand.xlsx.

Bash

python train.py
You will see the RMSE and MAE metrics printed to the console after training is complete.

Step 2: Generate Predictions This script will load the saved model from models/, make predictions, and save the results to outputs/predictions.csv.

Bash

python predict.py

3. (Optional) Exploratory Data Analysis (EDA)
To view the visualizations from the original notebook, run the eda.py script.

Bash

python eda.py

Note on models/ and outputs/ Folders: These folders are intentionally empty in the repository. The .gitignore file prevents output files (like .pkl, .csv, .xlsx) from being committed. These files are generated on your local machine when you run train.py and predict.py.


