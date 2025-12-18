# Fraud Detection System using Logistic Regression

## 📌 Project Overview
This project implements an end-to-end **Fraud Detection System** using **Logistic Regression** on a highly imbalanced real-world transaction dataset.  
The objective is to accurately identify fraudulent transactions while optimizing the **precision–recall tradeoff**, which is critical in fraud detection problems.

---

## 🔍 Key Features
- Handled **highly imbalanced data** (fraud ratio ≈ 1:773)
- Applied **feature preprocessing** using `ColumnTransformer`
- Used **class weighting and threshold tuning** instead of accuracy
- Optimized fraud-class **F1-score to 0.43**
- Deployed model using **Streamlit**
- Serialized model using **Joblib** for inference

---

## Machine Learning Details
- **Algorithm:** Logistic Regression
- **Evaluation Metrics:** Precision, Recall, F1-score, PR tradeoff
- **Imbalance Handling:** `class_weight='balanced'`
- **Threshold Optimization:** Custom probability threshold

---

## 📊 Model Performance (Fraud Class)
| Metric | Score |
|------|------|
| Precision | 0.33 |
| Recall | 0.60 |
| F1-score | 0.43 |

> Accuracy is intentionally not used as the primary metric due to extreme class imbalance.

---

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Joblib
- **Visualization:** Matplotlib
- **Deployment:** Streamlit
- **IDE:** Jupyter Notebook, VS Code

---

## 📁 Dataset
Due to GitHub size limitations, the dataset is **not included** in this repository.

📎 **Dataset Source (Kaggle):**  
👉 https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset

After downloading, place the CSV file inside the `data/` directory.

---

## 🚀 How to Run

### Install dependencies
pip install -r requirements.txt

### Run Streamlit App
streamlit run app.py

### Author

Eswarakumar J

LinkedIn: https://linkedin.com/in/eswarakumar-j

GitHub: https://github.com/EKtechie
