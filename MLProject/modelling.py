import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import os

# ── Konfigurasi MLflow ──────────────────────────────────────────────────────
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME", "nurulftriah")
DAGSHUB_REPO     = "Eksperimen_SML_Nurul-Fitriah"

mlflow.set_tracking_uri(
    f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow"
)
mlflow.set_experiment("credit-scoring-experiment")

# ── Load data preprocessing ─────────────────────────────────────────────────
DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "namadataset_preprocessing",
    "credit_scoring_preprocessing.csv"
)

df = pd.read_csv(DATA_PATH)

TARGET = "loan_status"
X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Hyperparameter tuning + logging ─────────────────────────────────────────
param_grid = {
    "n_estimators": [100, 200],
    "max_depth":    [5, 10, None],
    "min_samples_split": [2, 5],
}

with mlflow.start_run(run_name="RandomForest-GridSearch"):

    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("test_size",  0.2)
    mlflow.log_param("random_state", 42)

    rf   = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    mlflow.log_params(grid.best_params_)

    y_pred = best.predict(X_test)
    y_prob = best.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc",  auc)

    mlflow.sklearn.log_model(best, "model")

    print(f"✅ Training selesai")
    print(f"   Accuracy : {acc:.4f}")
    print(f"   F1 Score : {f1:.4f}")
    print(f"   ROC-AUC  : {auc:.4f}")
    print(f"   Best params: {grid.best_params_}")
