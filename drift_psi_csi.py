import pandas as pd
import numpy as np
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("loan_default_prediction")

# ======================
# LOAD DATA
# ======================
df = pd.read_csv("data/Loan_default.csv")

# simulate new data (for assignment)
df_new = df.sample(frac=0.5, random_state=42)
df_train = df.drop(df_new.index)

# ======================
# PSI FUNCTION (NUMERIC)
# ======================
def calculate_psi(expected, actual, bins=10):

    expected = np.array(expected)
    actual = np.array(actual)

    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))

    expected_counts = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_counts = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    # avoid zero division
    expected_counts = np.where(expected_counts == 0, 0.0001, expected_counts)
    actual_counts = np.where(actual_counts == 0, 0.0001, actual_counts)

    psi = np.sum((actual_counts - expected_counts) * np.log(actual_counts / expected_counts))

    return psi

# ======================
# CSI FUNCTION (CATEGORICAL)
# ======================
def calculate_csi(expected, actual):

    expected_counts = expected.value_counts(normalize=True)
    actual_counts = actual.value_counts(normalize=True)

    categories = set(expected_counts.index).union(set(actual_counts.index))

    csi = 0

    for cat in categories:
        e = expected_counts.get(cat, 0.0001)
        a = actual_counts.get(cat, 0.0001)

        csi += (a - e) * np.log(a / e)

    return csi

# ======================
# COLUMN SELECTION (FIXED)
# ======================

# numeric columns (exclude ID + target)
num_cols = df.select_dtypes(include=["int64", "float64"]).columns
num_cols = [col for col in num_cols if col not in ["LoanID", "Default"]]

# categorical columns (exclude ID)
cat_cols = df.select_dtypes(include=["object"]).columns
cat_cols = [col for col in cat_cols if col != "LoanID"]

# ======================
# CALCULATE DRIFT
# ======================

psi_results = {}
csi_results = {}

print("\n=== PSI (NUMERIC DRIFT) ===")

for col in num_cols:
    psi = calculate_psi(df_train[col], df_new[col])
    psi_results[col] = psi
    print(f"{col}: PSI = {psi:.4f}")

print("\n=== CSI (CATEGORICAL DRIFT) ===")

for col in cat_cols:
    csi = calculate_csi(df_train[col], df_new[col])
    csi_results[col] = csi
    print(f"{col}: CSI = {csi:.4f}")

# ======================
# LOG TO MLFLOW
# ======================

with mlflow.start_run(run_name="data_drift_psi_csi"):

    for col, psi in psi_results.items():
        mlflow.log_metric(f"psi_{col}", psi)

    for col, csi in csi_results.items():
        mlflow.log_metric(f"csi_{col}", csi)

print("\n✅ PSI + CSI logged to MLflow")