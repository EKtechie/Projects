# Fraud Detection System using XGBoost

🌐 **Live Demo:** [https://projects-fd.streamlit.app/](https://projects-fd.streamlit.app/)

## 📌 Project Overview
This project implements an end-to-end **Fraud Detection System** using **XGBoost** on a highly imbalanced financial transaction dataset.  
The objective is to accurately identify fraudulent transactions while optimizing the **precision–recall tradeoff**, which is critical in fraud detection problems.

---

## 🔍 Key Features
- Handled **highly imbalanced data** (fraud ratio ≈ 1:774 — only 8,213 fraud cases out of 6.3M transactions)
- Applied **custom feature preprocessing** using a `SimplePreprocessor` (StandardScaler + OneHotEncoder)
- Used **SMOTE** (Synthetic Minority Oversampling) to balance the training set
- Built an end-to-end **ImbPipeline**: Preprocessor → SMOTE → XGBClassifier
- Optimized fraud-class **F1-score to 0.86** via threshold tuning
- Deployed model using **Streamlit** — available live at [projects-fd.streamlit.app](https://projects-fd.streamlit.app/)
- Serialized model using **Joblib** for inference

---

## Machine Learning Details
- **Algorithm:** XGBoost (`XGBClassifier`)
- **Evaluation Metrics:** Precision, Recall, F1-score, PR Curve
- **Imbalance Handling:** SMOTE (via `imbalanced-learn`)
- **Threshold Optimization:** Best threshold = `0.9950` (maximizes F1)

---

## 📊 Model Performance (Fraud Class)
| Metric    | Score  |
| --------- | ------ |
| Precision | 0.92   |
| Recall    | 0.81   |
| F1-score  | 0.86   |
| Accuracy  | 99.78% |

**Confusion Matrix:**
|                    | Predicted: Legit | Predicted: Fraud |
|--------------------|-----------------|-----------------|
| **Actual: Legit**  | 1,906,144        | 178              |
| **Actual: Fraud**  | 478              | 1,986            |

> Accuracy alone is misleading due to extreme class imbalance — F1-score and the Precision-Recall curve are the primary evaluation metrics.

---

## 🛠️ Tech Stack
- **Language:** Python 3.12
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, imbalanced-learn, Joblib
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

### Option 1: Live App (No Setup Required!)
👉 Visit **[https://projects-fd.streamlit.app/](https://projects-fd.streamlit.app/)** directly in your browser — no installation needed.

---

### Option 2: Local Installation

#### Prerequisites
- Python 3.11 or 3.12 (recommended)

#### Step 1: Create and activate virtual environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

#### Step 3: Download the dataset
The dataset is **not included** in this repository due to GitHub size limitations.

📎 **Dataset Source (Kaggle):**  
👉 https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset

After downloading, place the CSV file inside the `data/` directory.

#### Step 4: Run the Streamlit App
```bash
streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`

---

### Troubleshooting
For detailed troubleshooting steps and solutions, see [SETUP.md](SETUP.md)

---

## 👤 Author
Eswarakumar J  
LinkedIn: https://linkedin.com/in/eswarakumar-j  
GitHub: https://github.com/EKtechie
