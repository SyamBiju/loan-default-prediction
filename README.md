# Loan Default Prediction System

## Overview
This project builds an end-to-end machine learning system to predict whether a customer will default on a loan. It covers the full ML lifecycle including data analysis, preprocessing, model training, experiment tracking, explainability, drift detection, and deployment.

---

## Problem Statement
Financial institutions need to assess the risk of loan default before approving loans. This project predicts loan default using customer financial and demographic data.

---

## Approach

### 1. Exploratory Data Analysis (EDA)
- Analyzed feature distributions and relationships
- Checked class imbalance in target variable
- Identified important numerical and categorical features

---

### 2. Data Preprocessing
- Removed irrelevant columns (LoanID)
- Separated numerical and categorical features
- Applied:
  - StandardScaler (numerical)
  - OneHotEncoder (categorical)
- Used ColumnTransformer inside a Pipeline

---

### 3. Model Development
Trained multiple models:
- Logistic Regression
- Random Forest (with hyperparameter tuning)

### Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

### 4. MLflow Tracking
- Logged experiments, parameters, and metrics
- Compared multiple model runs
- Stored best model
- Logged:
  - Performance metrics
  - Hyperparameters
  - SHAP artifacts
  - Drift metrics (KS, PSI, CSI)

---

### 5. Explainability (XAI - SHAP)
- Used SHAP TreeExplainer for interpretation
- Generated:
  - SHAP summary plot
  - SHAP feature importance (bar plot)
- Logged plots in MLflow

---

### 6. Data Drift Detection

#### KS Test
- Compared distributions of train vs new data
- Identified feature-level drift

#### PSI (Population Stability Index)
- Used for numerical features
- Thresholds:
  - < 0.1 → No drift
  - 0.1–0.25 → Moderate drift
  - > 0.25 → Significant drift

#### CSI (Characteristic Stability Index)
- Used for categorical features

All drift metrics were logged in MLflow.

---

### 7. Deployment (Docker)

- Built a REST API using Flask
- Containerized using Docker
- Exposed endpoint for predictions

---

## How to Run

### 1. Train Model
```bash
python model.py

```
### 2. Run MLflow UI
```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db

```
### Open in browser:
http://127.0.0.1:5000


### Run SHAP

```bash
python explain.py

```
### Run Drift Detection

```bash
python drift.py
python drift_psi_csi.py

```
## Run Docker
### Build image:
```bash
docker build -t loan-default-app .

```
### Run container:
```bash
docker run -p 5000:5000 loan-default-app

```
### Test API (PowerShell):

```bash 
Invoke-RestMethod -Uri http://localhost:5000/predict `
-Method POST `
-ContentType "application/json" `
-Body '{
  "Age":35,
  "Income":50000,
  "LoanAmount":20000,
  "CreditScore":700,
  "MonthsEmployed":24,
  "NumCreditLines":3,
  "InterestRate":10,
  "LoanTerm":36,
  "DTIRatio":0.3,
  "Education":"Bachelor''s",
  "EmploymentType":"Full-time",
  "MaritalStatus":"Single",
  "HasMortgage":"No",
  "HasDependents":"No",
  "LoanPurpose":"Auto",
  "HasCoSigner":"No"
}'

```

## Project Structure

.
├── data/
├── models/
├── model.py
├── explain.py
├── drift.py
├── drift_psi_csi.py
├── app.py
├── Dockerfile
├── requirements.txt
├── mlflow.db
└── mlruns/

---
## Key Insights
Income and Age were the most influential features.
Minimal data drift observed in current dataset.
Random Forest performed better than Logistic Regression.

---

## Conclusion

This project demonstrates a complete machine learning pipeline from data preprocessing to deployment. It integrates explainability and monitoring, making it suitable for real-world production scenarios.

---

### Author

Syam B





