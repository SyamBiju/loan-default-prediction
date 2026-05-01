import shap
import pandas as pd
import joblib
import mlflow
import matplotlib.pyplot as plt
import numpy as np

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("loan_default_prediction")

# Load model
model = joblib.load("models/model.pkl")

# Load data
df = pd.read_csv("data/Loan_default.csv")

X = df.drop(columns=["LoanID", "Default"])

# sample
X_sample = X.sample(50, random_state=42)

# pipeline parts
preprocessor = model.named_steps["prep"]
rf_model = model.named_steps["model"]

# transform
X_transformed = preprocessor.transform(X_sample)
feature_names = preprocessor.get_feature_names_out()

X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)

# SHAP (Tree Explainer)
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_transformed_df)

# classification handling
if isinstance(shap_values, list):
    shap_to_plot = shap_values[1]
else:
    shap_to_plot = shap_values

shap_to_plot = np.array(shap_to_plot)

# ======================
# SUMMARY PLOT
# ======================
plt.figure()
shap.summary_plot(shap_to_plot, X_transformed_df, show=False)
plt.savefig("shap_summary.png", bbox_inches="tight")
plt.close()

# ======================
# BAR PLOT
# ======================
plt.figure()
shap.summary_plot(shap_to_plot, X_transformed_df, plot_type="bar", show=False)
plt.savefig("shap_bar.png", bbox_inches="tight")
plt.close()

# ======================
# LOG TO MLFLOW
# ======================
with mlflow.start_run(run_name="shap_explainability"):
    mlflow.log_artifact("shap_summary.png")
    mlflow.log_artifact("shap_bar.png")

print("✅ SHAP successfully logged to MLflow")