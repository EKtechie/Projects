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

| Metric    | Score |
| --------- | ----- |
| Precision | 0.33  |
| Recall    | 0.60  |
| F1-score  | 0.43  |

> Accuracy is intentionally not used as the primary metric due to extreme class imbalance.

---

## 🛠️ Tech Stack

- **Language:** Python 3.12
- **Libraries:** Pandas, NumPy, Scikit-learn (1.6.1), Joblib
- **Visualization:** Matplotlib
- **Deployment:** Streamlit
- **Containerization:** Docker & Docker Compose
- **IDE:** Jupyter Notebook, VS Code

---

## 📁 Dataset

Due to GitHub size limitations, the dataset is **not included** in this repository.

📎 **Dataset Source (Kaggle):**  
👉 https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset

After downloading, place the CSV file inside the `data/` directory.

---

## 🚀 How to Run

### Option 1: Docker (Recommended - No Environment Issues!)

The easiest way to run the app without any dependency conflicts.

#### Prerequisites

- **Docker:** [Download Docker Desktop](https://www.docker.com/products/docker-desktop)

#### Quick Start

```bash
docker-compose up
```

The app will automatically start at: `http://localhost:8501`

**Benefits:**

- ✅ No Python version conflicts
- ✅ All dependencies pre-configured
- ✅ Works on Windows, Mac, and Linux
- ✅ No virtual environment needed

To stop: `docker-compose down`

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
