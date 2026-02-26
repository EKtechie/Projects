# Setup Guide - Fraud Detection System

This guide ensures the project runs smoothly on **Windows**, **Mac**, and **Linux** machines.

## ✅ System Requirements

- **Python:** 3.11 or 3.12 (recommended for package compatibility)
- **pip:** Latest version

## 🔧 Installation Steps

### 1. Clone or Download the Repository

```bash
git clone <repository-url>
cd "Fraud Detection Log Regression"
```

### 2. Create a Virtual Environment (Recommended)

**On Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**On Mac/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Upgrade pip

```bash
pip install --upgrade pip
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Download the Dataset

1. Visit [Kaggle - Fraud Detection Dataset](https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset)
2. Download the CSV file
3. Place it in the `data/` folder

Your structure should look like:

```
Fraud Detection Log Regression/
├── data/
│   └── fraud_detection.csv  (or whatever the filename is)
├── model/
│   └── Fraud_Detection_Log_model.pkl
├── app.py
├── requirements.txt
└── ...
```

### 6. Run the Application

```bash
streamlit run app.py
```

The app will open in your default browser at: `http://localhost:8501`

---

## 🐛 Troubleshooting

### Issue: "AttributeError: module 'sklearn.compose.\_column_transformer' has no attribute '\_RemainderColsList'"

**Cause:** scikit-learn version incompatibility with the pickled model, often due to using Python 3.13+ which doesn't have wheel distributions for compatible scikit-learn versions.

**Best Solution:** Use **Python 3.11 or 3.12**:

1. Create a new virtual environment with Python 3.11 or 3.12
2. Activate it
3. Run: `pip install -r requirements.txt`

**Alternative (if you must use Python 3.13+):**

Uninstall scikit-learn and install the latest version:

**On Windows (Command Prompt):**

```bash
pip uninstall scikit-learn -y
pip install --only-binary=:all: scikit-learn
```

**On Mac/Linux:**

```bash
pip uninstall scikit-learn -y
pip install scikit-learn
```

Then restart your Streamlit app:

```bash
streamlit run app.py
```

If the model still fails to load, you will need to **retrain the model** with the new scikit-learn version.

---

### Issue: "ModuleNotFoundError" when running the app

**Solution:** Make sure your virtual environment is activated:

- Windows: `venv\Scripts\activate`
- Mac/Linux: `source venv/bin/activate`

### Issue: "Fraud_Detection_Log_model.pkl not found"

**Solution:** Ensure the model file exists in the `model/` directory (it should be pre-included)

### Issue: Dataset not found when training

**Solution:** Download the dataset from Kaggle and place it in the `data/` folder

### Issue: Permission denied on Mac/Linux

**Solution:** Make sure you have read/write permissions in the project directory

---

## 📦 Version Compatibility

The project works best with **Python 3.11 or 3.12**:

- scikit-learn >= 1.3.0, < 1.5.0 (for Python 3.11-3.12)
- pandas >= 1.0.0
- numpy >= 1.19.0
- joblib >= 1.3.0
- streamlit >= 1.0.0

**Note on Python 3.13+:** Older scikit-learn versions don't have wheel distributions for Python 3.13+. If using Python 3.13+, install the latest scikit-learn and retrain the model, or use Python 3.11/3.12 instead.

---

## 💡 Tips

- Use a virtual environment to avoid version conflicts with other projects
- Keep dependencies updated: `pip install --upgrade -r requirements.txt`
- On slower connections, you may need to increase the pip timeout: `pip install --default-timeout=1000 -r requirements.txt`

---

## � Docker Setup (Recommended for Avoiding Environment Issues)

Using Docker eliminates all dependency and environment compatibility issues. The app will run consistently on any machine with Docker installed.

### Prerequisites

- **Docker:** [Download and Install Docker](https://www.docker.com/products/docker-desktop)
- **Docker Compose:** Usually comes with Docker Desktop

### Quick Start with Docker Compose

The easiest way to run the app:

```bash
docker-compose up
```

The app will automatically build and start at: `http://localhost:8501`

To stop:

```bash
docker-compose down
```

### Building and Running with Docker Directly

**Build the image:**

```bash
docker build -t fraud-detection-app .
```

**Run the container:**

```bash
docker run -p 8501:8501 -v $(pwd)/data:/app/data -v $(pwd)/model:/app/model fraud-detection-app
```

On Windows (Command Prompt):

```bash
docker run -p 8501:8501 -v %CD%\data:/app/data -v %CD%\model:/app/model fraud-detection-app
```

### Docker Benefits

- ✅ No Python version conflicts
- ✅ All dependencies automatically handled
- ✅ Consistent environment across Windows, Mac, and Linux
- ✅ Easy to share and deploy
- ✅ No need for virtual environments

---

## �📞 Support

If you encounter any issues, please check the main README.md or contact the project author.
