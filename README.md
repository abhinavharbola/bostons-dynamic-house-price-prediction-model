# bostonHD - Boston House Price Dynamics

**bostonHD** is a sleek, two-page Streamlit web app that predicts **Boston house prices** using a **Random Forest Regressor**.  
An interactive interface for real-time predictions and a SHAP-based explanation page for model interpretability.

---

## Features

- **Interactive UI** — Adjust housing parameters with sliders and tooltips.  
- **Production-Ready Architecture** — Decoupled *training* (`train.py`) and *inference* (`home.py`) pipelines.  
- **Explainable AI (XAI)** — SHAP visualizations for model interpretability.  
- **Proper Validation** — Reports *MAE* and *R²* on the test set.  
- **Clean Separation** — Offline model training, online inference.  
- **Ethical Dataset** — The racially biased “B” feature has been removed.  
- **No Data Leakage** — UI ranges are generated from training set statistics only.  

---

## Tech Stack

| Component        | Technology                  |
|------------------|-----------------------------|
| **Frontend**     | Streamlit                   |
| **Backend/Model**| scikit-learn, Joblib        |
| **Data**         | Pandas, PyArrow             |
| **Explainability** | SHAP                      |
| **Language**     | Python 3.8+                 |

---

## Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/boston-house-price-predictor.git
cd boston-house-price-predictor
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
# macOS/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> The `requirements.txt` includes:
> - `streamlit`
> - `scikit-learn`
> - `joblib`
> - `pyarrow`
> - `pandas`
> - `shap`

---

## Train the Model (Offline Step)

Run the training script once to generate the artifacts required for inference:

```bash
python train.py
```

This will:
- Train a `RandomForestRegressor`  
- Validate it (prints MAE, R²)  
- Save:
  - `model.joblib` — trained model  
  - `model_artifacts.json` — feature stats & UI info  
  - `X_train.parquet` — background data for SHAP  

---

## Run the Streamlit App

Once artifacts are ready, launch the app:

```bash
streamlit run home.py
```

The app will:
- Load the pre-trained model and metadata  
- Build dynamic UI sliders from `model_artifacts.json`  
- Predict and display housing prices in real-time  

---

## How It Works

### **Offline Training (`train.py`)**
1. Loads Boston Housing dataset (with “B” feature removed).  
2. Splits into train/test sets (80/20).  
3. Trains a Random Forest model.  
4. Evaluates and prints MAE, R².  
5. Saves all training artifacts.  

### **Online Inference (`home.py`)**
1. Loads the saved model and metadata.  
2. Dynamically generates sliders from training statistics.  
3. Captures user input into a DataFrame.  
4. Predicts and displays median house value.  

---

## Project Structure

```
boston-house-price-predictor/
│
├── home.py                # Main Streamlit app (inference)
├── train.py               # Offline training script
│
├── pages/
│   └── shap_graph.py      # SHAP-based explainability page
│
├── model.joblib           # Trained scikit-learn model
├── model_artifacts.json   # Feature stats, UI info
├── X_train.parquet        # Background data for SHAP
│
├── requirements.txt       # Dependencies
└── README.md              # Documentation
```

---

## Key Features Explained

| Feature | Description |
|----------|-------------|
| **CRIM** | Per capita crime rate by town |
| **ZN** | Proportion of residential land zoned for large lots |
| **INDUS** | Non-retail business acres per town |
| **CHAS** | Charles River dummy variable (1 if tract bounds river) |
| **NOX** | Nitric oxides concentration (ppm) |
| **RM** | Average number of rooms per dwelling |
| **AGE** | Proportion of owner-occupied units built before 1940 |
| **DIS** | Weighted distances to Boston employment centers |
| **RAD** | Accessibility to radial highways |
| **TAX** | Property-tax rate per $10,000 |
| **PTRATIO** | Pupil–teacher ratio by town |
| **LSTAT** | % lower status of the population |

---


**Author:** [Abhinav Harbola](https://github.com/abhinavharbola)
