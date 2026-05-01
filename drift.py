import pandas as pd
import mlflow
from scipy.stats import ks_2samp

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("loan_default_prediction")

# Load data
df = pd.read_csv("data/Loan_default.csv")

train = df.sample(frac=0.7, random_state=42)
new_data = df.drop(train.index)

cols = ["Age", "Income", "LoanAmount", "CreditScore"]

with mlflow.start_run(run_name="data_drift_ks"):

    print("\n=== KS DRIFT REPORT ===\n")

    for col in cols:
        stat, p_value = ks_2samp(train[col], new_data[col])

        print(f"{col}: KS p-value = {p_value:.4f}")

        mlflow.log_metric(f"ks_pvalue_{col}", p_value)

        if p_value < 0.05:
            print(" ⚠️ Drift Detected\n")
        else:
            print(" ✅ No Drift\n")