
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "sensor_data.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

FEATURES = [
    "vibration",
    "temperature",
    "voltage",
    "acoustic_noise"
]

X = df[FEATURES]
y_rul = df["rul_hours"]
y_failure = df["failure_24h"]

# Use the same split for both models
X_train, X_test, y_rul_train, y_rul_test = train_test_split(
    X, y_rul, test_size=0.2, random_state=42
)

# Reproduce the same split for classification
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_failure, test_size=0.2, random_state=42, stratify=y_failure
)

# 1. RUL regression pipeline
rul_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

rul_pipeline.fit(X_train, y_rul_train)
rul_pred = rul_pipeline.predict(X_test)

print("=" * 50)
print("REMAINING USEFUL LIFE (RUL) PREDICTION")
print("=" * 50)
print(f"MAE:  {mean_absolute_error(y_rul_test, rul_pred):.2f} hours")
print(f"RMSE: {root_mean_squared_error(y_rul_test, rul_pred):.2f} hours")
print(f"R2:   {r2_score(y_rul_test, rul_pred):.4f}")

# 2. Failure classification pipeline
failure_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

failure_pipeline.fit(X_train_c, y_train_c)

# Choose the default 0.5 threshold for this baseline evaluation
failure_probs = failure_pipeline.predict_proba(X_test_c)[:, 1]
failure_pred = (failure_probs >= 0.5).astype(int)

print("\n" + "=" * 50)
print("FAILURE CLASSIFICATION - THRESHOLD 0.5")
print("=" * 50)
print(f"Accuracy:  {accuracy_score(y_test_c, failure_pred):.3f}")
print(f"Precision: {precision_score(y_test_c, failure_pred, zero_division=0):.3f}")
print(f"Recall:    {recall_score(y_test_c, failure_pred, zero_division=0):.3f}")
print(f"F1 score:  {f1_score(y_test_c, failure_pred, zero_division=0):.3f}")
print("Confusion matrix:")
print(confusion_matrix(y_test_c, failure_pred))

# Save both fitted pipelines
joblib.dump(rul_pipeline, MODEL_DIR / "rul_pipeline.joblib")
joblib.dump(failure_pipeline, MODEL_DIR / "failure_pipeline.joblib")

print("\nSaved models to:", MODEL_DIR)
