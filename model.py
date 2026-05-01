import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("loan_default_prediction")

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv("data/Loan_default.csv")

# Drop ID
df = df.drop("LoanID", axis=1)

# Features / Target
X = df.drop("Default", axis=1)
y = df["Default"]

# Column types
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

# Preprocessing
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

best_model = None
best_score = 0

# =========================
# 1. Logistic Regression
# =========================
with mlflow.start_run(run_name="logistic_regression"):

    log_model = LogisticRegression(max_iter=1000, class_weight="balanced")

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", log_model)
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    mlflow.log_param("model_name", "logistic_regression")
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", auc)

    mlflow.sklearn.log_model(pipeline, "logistic_regression")

    if auc > best_score:
        best_score = auc
        best_model = pipeline

# =========================
# 2. Random Forest (Hyperparameter Experiments)
# =========================

rf_params_list = [
    {"n_estimators": 100, "max_depth": 10},
    {"n_estimators": 200, "max_depth": 10},
    {"n_estimators": 200, "max_depth": 20}
]

for params in rf_params_list:

    run_name = f"rf_{params['n_estimators']}_{params['max_depth']}"

    with mlflow.start_run(run_name=run_name):

        rf_model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            class_weight="balanced",
            random_state=42
        )

        pipeline = Pipeline([
            ("prep", preprocessor),
            ("model", rf_model)
        ])

        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)
        probs = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        # Log parameters
        mlflow.log_param("model_name", "random_forest")
        mlflow.log_param("n_estimators", params["n_estimators"])
        mlflow.log_param("max_depth", params["max_depth"])

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        mlflow.sklearn.log_model(pipeline, run_name)

        print(f"\nRandom Forest Params: {params}")
        print("Accuracy:", acc)
        print("F1 Score:", f1)
        print("ROC AUC:", auc)

        if auc > best_score:
            best_score = auc
            best_model = pipeline

# Save best model
joblib.dump(best_model, "models/model.pkl")
print("\nBest model saved.")