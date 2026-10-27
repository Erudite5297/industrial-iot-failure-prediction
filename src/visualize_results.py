
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "sensor_data.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "reports" / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset and saved models
df = pd.read_csv(DATA_PATH)
rul_model = joblib.load(MODEL_DIR / "rul_pipeline.joblib")
failure_model = joblib.load(MODEL_DIR / "failure_pipeline.joblib")

FEATURES = [
    "vibration",
    "temperature",
    "voltage",
    "acoustic_noise"
]

# Recreate the same test split used in train_pipeline.py
X = df[FEATURES]
y_rul = df["rul_hours"]
y_failure = df["failure_24h"]

X_train, X_test, y_rul_train, y_rul_test = train_test_split(
    X, y_rul, test_size=0.2, random_state=42
)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_failure, test_size=0.2, random_state=42, stratify=y_failure
)

# ---------------------------------
# 1. Actual vs predicted RUL
# ---------------------------------
rul_predictions = rul_model.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(y_rul_test, rul_predictions, alpha=0.6)
plt.xlabel("Actual RUL (hours)")
plt.ylabel("Predicted RUL (hours)")
plt.title("Actual vs Predicted Remaining Useful Life")

# Reference line for perfect predictions
low = min(y_rul_test.min(), rul_predictions.min())
high = max(y_rul_test.max(), rul_predictions.max())
plt.plot([low, high], [low, high], linestyle="--")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "rul_actual_vs_predicted.png", dpi=150)
plt.close()

# ---------------------------------
# 2. Sensor feature distributions
# ---------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 7))

for column, ax in zip(FEATURES, axes.flatten()):
    sns.histplot(data=df, x=column, kde=True, ax=ax)
    ax.set_title(f"Distribution of {column}")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sensor_distributions.png", dpi=150)
plt.close()

# ---------------------------------
# 3. Failure confusion matrix
# ---------------------------------
failure_predictions = failure_model.predict(X_test_c)
cm = confusion_matrix(y_test_c, failure_predictions)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal (0)", "Failure (1)"],
    yticklabels=["Normal (0)", "Failure (1)"]
)
plt.xlabel("Predicted label")
plt.ylabel("Actual label")
plt.title("Failure Classification Confusion Matrix")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "failure_confusion_matrix.png", dpi=150)
plt.close()

print("All visualizations generated successfully!")
print(f"Saved in: {OUTPUT_DIR}")
print("\nGenerated files:")
for file in sorted(OUTPUT_DIR.glob("*.png")):
    print(f"- {file.name}")
