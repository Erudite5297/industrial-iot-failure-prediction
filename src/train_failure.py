
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Load dataset
root = Path(__file__).resolve().parent.parent
df = pd.read_csv(root / "data" / "sensor_data.csv")

X = df[[
    "vibration",
    "temperature",
    "voltage",
    "acoustic_noise"
]]
y = df["failure_24h"]

# 60% training, 20% validation, 20% testing
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Train model using training data only
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Get probabilities for the positive class (failure = 1)
val_probs = model.predict_proba(X_val)[:, 1]

# Compare thresholds on validation data
print("THRESHOLD TUNING ON VALIDATION DATA")
print("-" * 45)
print(f"{'Threshold':<12}{'Precision':<12}{'Recall':<12}{'F1':<10}")

for threshold in [0.2, 0.3, 0.4, 0.5, 0.6]:
    val_pred = (val_probs >= threshold).astype(int)

    precision = precision_score(
        y_val, val_pred, zero_division=0
    )
    recall = recall_score(
        y_val, val_pred, zero_division=0
    )
    f1 = f1_score(y_val, val_pred, zero_division=0)

    print(
        f"{threshold:<12.1f}{precision:<12.3f}"
        f"{recall:<12.3f}{f1:<10.3f}"
    )

# Select threshold with recall of at least 0.80,
# if any candidate achieves it; otherwise choose
# the candidate with the highest validation recall.
thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]
results = []

for threshold in thresholds:
    pred = (val_probs >= threshold).astype(int)
    results.append({
        "threshold": threshold,
        "recall": recall_score(y_val, pred, zero_division=0),
        "precision": precision_score(y_val, pred, zero_division=0)
    })

eligible = [r for r in results if r["recall"] >= 0.80]

if eligible:
    best = max(eligible, key=lambda r: r["precision"])
else:
    best = max(results, key=lambda r: r["recall"])

chosen_threshold = best["threshold"]

# Evaluate once on the untouched test set
test_probs = model.predict_proba(X_test)[:, 1]
test_pred = (test_probs >= chosen_threshold).astype(int)

print("\nFINAL TEST RESULTS")
print("-" * 45)
print(f"Chosen threshold: {chosen_threshold:.2f}")
print(f"Accuracy:  {accuracy_score(y_test, test_pred):.3f}")
print(f"Precision: {precision_score(y_test, test_pred, zero_division=0):.3f}")
print(f"Recall:    {recall_score(y_test, test_pred, zero_division=0):.3f}")
print(f"F1 Score:  {f1_score(y_test, test_pred, zero_division=0):.3f}")
print("Confusion matrix:")
print(confusion_matrix(y_test, test_pred))
