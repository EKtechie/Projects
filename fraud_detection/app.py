import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# --- Define Preprocessor (must match training exactly) ---
class SimplePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, numerical, categorical):
        self.numerical = numerical
        self.categorical = categorical
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    def fit(self, X, y=None):
        self.scaler.fit(X[self.numerical])
        self.encoder.fit(X[self.categorical])
        return self

    def transform(self, X):
        X_num = self.scaler.transform(X[self.numerical])
        X_cat = self.encoder.transform(X[self.categorical])
        return np.hstack([X_num, X_cat])

# Required so joblib can unpickle the custom class
import sys
setattr(sys.modules[__name__], 'SimplePreprocessor', SimplePreprocessor)

# --- Page Config ---
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Financial Transaction Fraud Detection")
st.markdown("Predict whether a transaction is potentially fraudulent using an XGBoost model trained with SMOTE balancing.")

# --- Load Model ---
@st.cache_resource
def load_model():
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "Fraud_Detection_XGB_model.pkl")
    return joblib.load(model_path)

try:
    model = load_model()
    st.success("✅ Model loaded successfully!")
except ModuleNotFoundError as e:
    st.error(f"❌ Missing dependency: `{e.name}`")
    st.info(
        "This model requires additional packages. Run the following in your terminal:\n\n"
        "```\npip install imbalanced-learn xgboost\n```\n\n"
        "Then restart the Streamlit app."
    )
    st.stop()
except FileNotFoundError:
    st.error("❌ Model file not found at `models/Fraud_Detection_XGB_model.pkl`")
    st.info("Make sure the `models/` folder is in the same directory as `app.py`.")
    st.stop()
except Exception as e:
    st.error(f"❌ Could not load model: {e}")
    st.stop()

# --- User Input ---
st.header("Transaction Details")

col1, col2 = st.columns(2)

with col1:
    tx_type = st.selectbox(
        "Transaction Type",
        options=["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"],
        help="Select the type of transaction."
    )
    amount = st.number_input(
        "Transaction Amount ($)",
        min_value=0.0,
        value=1000.0,
        step=100.0,
        help="The amount of the transaction."
    )

with col2:
    oldbalanceOrg = st.number_input(
        "Sender's Initial Balance ($)",
        min_value=0.0,
        value=5000.0,
        step=500.0
    )
    newbalanceOrig = st.number_input(
        "Sender's New Balance ($)",
        min_value=0.0,
        value=4000.0,
        step=500.0
    )

col3, col4 = st.columns(2)
with col3:
    oldbalanceDest = st.number_input(
        "Receiver's Initial Balance ($)",
        min_value=0.0,
        value=0.0,
        step=500.0
    )
with col4:
    newbalanceDest = st.number_input(
        "Receiver's New Balance ($)",
        min_value=0.0,
        value=1000.0,
        step=500.0
    )

st.markdown("---")

# --- Prediction ---
if st.button("🔍 Detect Fraud", type="primary", use_container_width=True):
    input_data = pd.DataFrame([{
        "type": tx_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }])

    with st.spinner("Analyzing transaction..."):
        try:
            prob = model.predict_proba(input_data)[0, 1]
            pred = model.predict(input_data)[0]

            st.subheader("Result:")
            if pred == 1:
                st.error("🚨 **FRAUDULENT TRANSACTION DETECTED**")
                st.warning(f"Confidence (Fraud Probability): {prob:.2%}")
            else:
                st.success("✅ **TRANSACTION SEEMS LEGITIMATE**")
                st.info(f"Fraud Probability: {prob:.2%}")

        except Exception as e:
            st.error(f"Prediction error: {e}")
