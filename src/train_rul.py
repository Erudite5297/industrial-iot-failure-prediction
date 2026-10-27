
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Find and load the dataset
project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "sensor_data.csv"

df = pd.read_csv(data_path)

# Input features
X = df[[
    "vibration",
    "temperature",
    "voltage",
    "acoustic_noise"
]]

# Target variable
y = df["rul_hours"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict remaining useful life
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("LINEAR REGRESSION - RUL PREDICTION")
print("-" * 40)
print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")
print(f"MAE:  {mae:.2f} hours")
print(f"RMSE: {rmse:.2f} hours")
print(f"R2 Score: {r2:.4f}")

# Predict for a sample machine
sample = pd.DataFrame([{
    "vibration": 5.5,
    "temperature": 70.0,
    "voltage": 220.0,
    "acoustic_noise": 60.0
}])

predicted_rul = model.predict(sample)[0]

print("\nSample machine prediction:")
print(f"Estimated remaining useful life: {predicted_rul:.2f} hours")
