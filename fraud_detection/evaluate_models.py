import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, average_precision_score, roc_auc_score, f1_score

import joblib

# Load data
print("Loading data...")
df = pd.read_csv("data/AIML Dataset.csv")

# Preprocessing
print("Preprocessing...")
df_model = df.drop(["nameOrig","nameDest","isFlaggedFraud"], axis=1)
categorical = ["type"]
numerical = ["oldbalanceOrg","amount","newbalanceOrig","oldbalanceDest","newbalanceDest"]

X = df_model.drop("isFraud", axis=1)
y = df_model["isFraud"]

print(f"Total data shape: {X.shape}")
print(f"Total frauds: {y.sum()}")

# To accelerate the evaluation across multiple models, use 10% of the dataset while preserving class distribution
_, X_subset, _, y_subset = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
print(f"Using subset of data: {X_subset.shape}, subset frauds: {y_subset.sum()}")

X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, random_state=42, test_size=0.3, stratify=y_subset)

class SimplePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, numerical, categorical):
        self.numerical = numerical
        self.categorical=categorical
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

pipelines = {
    "Logistic Regression": Pipeline([
        ("prep", SimplePreprocessor(numerical, categorical)),
        ("clf", LogisticRegression(max_iter=1000, class_weight='balanced'))
    ]),
    "Random Forest": Pipeline([
        ("prep", SimplePreprocessor(numerical, categorical)),
        ("clf", RandomForestClassifier(n_estimators=50, n_jobs=-1, class_weight='balanced', random_state=42))
    ]),
    "Gradient Boosting": Pipeline([
        ("prep", SimplePreprocessor(numerical, categorical)),
        ("clf", GradientBoostingClassifier(n_estimators=50, random_state=42))
    ]),
    "Isolation Forest": Pipeline([
        ("prep", SimplePreprocessor(numerical, categorical)),
        ("clf", IsolationForest(contamination=0.0013, random_state=42, n_jobs=-1))
    ])
}

results = {}

print("Training models...")
for name, pipe in pipelines.items():
    print(f"\n==========================================")
    print(f"Evaluating {name}...")
    try:
        if name == "Isolation Forest":
            pipe.fit(X_train)
            preds = pipe.predict(X_test)
            y_pred = np.where(preds == -1, 1, 0)
            
            pr_auc = '-'
            roc_auc = '-'
            f1 = f1_score(y_test, y_pred)
        else:
            pipe.fit(X_train, y_train)
            probs = pipe.predict_proba(X_test)[:, 1]
            y_pred = pipe.predict(X_test)
            
            pr_auc = average_precision_score(y_test, probs)
            roc_auc = roc_auc_score(y_test, probs)
            f1 = f1_score(y_test, y_pred)
            
        print(classification_report(y_test, y_pred))
        
        if name != "Isolation Forest":
            print(f"PR AUC: {pr_auc:.4f}")
            print(f"ROC AUC: {roc_auc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        results[name] = {
            'PR_AUC': pr_auc,
            'ROC_AUC': roc_auc,
            'F1_Score': f1
        }
        
        # Save the model
        model_filename = f"models/{name.replace(' ', '_')}.pkl"
        joblib.dump(pipe, model_filename)
        print(f"Saved model to {model_filename}")
        
    except Exception as e:
        print(f"Error evaluating {name}: {e}")

print("\n--- Final Results Summary ---")
for name, metrics in results.items():
    pr = metrics['PR_AUC'] if isinstance(metrics['PR_AUC'], str) else f"{metrics['PR_AUC']:.4f}"
    roc = metrics['ROC_AUC'] if isinstance(metrics['ROC_AUC'], str) else f"{metrics['ROC_AUC']:.4f}"
    f1 = metrics['F1_Score']
    print(f"{name}: PR AUC={pr}, ROC AUC={roc}, F1={f1:.4f}")
