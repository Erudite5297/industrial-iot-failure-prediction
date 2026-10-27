
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

n = 2000

# Generate simulated sensor readings
vibration = np.random.normal(5, 1.2, n).clip(0.5, 10)
temperature = np.random.normal(70, 10, n).clip(30, 120)
voltage = np.random.normal(220, 8, n).clip(180, 260)
acoustic_noise = np.random.normal(60, 8, n).clip(30, 100)

# Estimate remaining useful life
rul = (
    200
    - 12 * vibration
    - 0.8 * temperature
    - 0.2 * acoustic_noise
    + 0.15 * voltage
    + np.random.normal(0, 10, n)
).clip(0, 200)

# Simulate failure risk within 24 hours
risk_score = (
    0.8 * vibration
    + 0.04 * temperature
    + 0.02 * acoustic_noise
    - 0.015 * voltage
)

failure_probability = 1 / (1 + np.exp(-(risk_score - 6)))
failure_24h = np.random.binomial(1, failure_probability)

# Create the dataset
df = pd.DataFrame({
    "vibration": vibration,
    "temperature": temperature,
    "voltage": voltage,
    "acoustic_noise": acoustic_noise,
    "rul_hours": rul,
    "failure_24h": failure_24h
})

# Save the dataset
project_root = Path(__file__).resolve().parent.parent
output_path = project_root / "data" / "sensor_data.csv"
df.to_csv(output_path, index=False)

print(f"Dataset saved to: {output_path}")
print(f"Dataset shape: {df.shape}")
print(df.head())
print("\nFailure label distribution:")
print(df["failure_24h"].value_counts())
