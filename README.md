
# Industrial IoT Telemetry Anomaly Detection & Failure Prediction

## 1. Project Overview

This project uses simulated industrial sensor data to predict the Remaining Useful Life (RUL) of a machine and estimate whether it may experience a failure within the next 24 hours.

The goal is to help maintenance teams identify potentially failing machines and plan maintenance before a breakdown occurs.

## 2. Objectives

- Generate industrial sensor telemetry data.
- Predict the Remaining Useful Life (RUL) of a machine.
- Predict the probability of failure within 24 hours.
- Evaluate model performance using appropriate metrics.
- Visualize model results.
- Save trained machine learning pipelines for reuse.

## 3. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Joblib
- Git and GitHub

## 4. Dataset

The project uses simulated sensor data with 2,000 records.

Features:

| Feature | Description |
|---|---|
| vibration | Machine vibration measurement |
| temperature | Machine temperature |
| voltage | Machine voltage measurement |
| acoustic_noise | Machine acoustic noise measurement |
| rul_hours | Remaining useful life in hours |
| failure_24h | Whether failure is expected within 24 hours |

The dataset is synthetically generated for learning and experimentation. It is not real industrial telemetry.

## 5. Machine Learning Models

### Remaining Useful Life Prediction

A Linear Regression model predicts the remaining useful life of a machine using sensor measurements.

Evaluation metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

### Failure Prediction

A Logistic Regression model estimates the probability of failure within 24 hours.

Evaluation metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

A separate experiment evaluates different classification thresholds to understand the trade-off between identifying failures and generating false alarms.

## 6. Results

### RUL Prediction

| Metric | Result |
|---|---:|
| MAE | 7.98 hours |
| RMSE | 9.89 hours |
| R² Score | 0.7270 |

### Failure Prediction — Baseline Threshold

The saved baseline classification pipeline uses a probability threshold of 0.50.

| Metric | Result |
|---|---:|
| Accuracy | 0.730 |
| Precision | 0.522 |
| Recall | 0.324 |
| F1-score | 0.400 |

### Failure Prediction — Threshold-Tuning Experiment

A separate experiment selected a threshold of 0.20 using validation data.

| Metric | Result |
|---|---:|
| Accuracy | 0.580 |
| Precision | 0.380 |
| Recall | 0.836 |
| F1-score | 0.523 |

The lower threshold identifies more actual failures but also produces more false alarms. The tuned-threshold results are from a separate experiment and should not be treated as a direct comparison against the baseline test results.

## 7. Visualizations

The project generates the following plots:

- `reports/figures/rul_actual_vs_predicted.png`
- `reports/figures/failure_confusion_matrix.png`
- `reports/figures/sensor_distributions.png`

These plots help evaluate prediction quality, classification errors, and sensor distributions.

## 8. Project Structure

```text
industrial-iot-failure-prediction/
├── data/
│   └── sensor_data.csv
├── src/
│   ├── generate_data.py
│   ├── train_rul.py
│   ├── train_failure.py
│   ├── train_pipeline.py
│   └── visualize_results.py
├── models/
│   ├── rul_pipeline.joblib
│   └── failure_pipeline.joblib
├── reports/
│   └── figures/
├── requirements.txt
└── README.md
```

## 9. Setup and Execution

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate the dataset:

```bash
python src/generate_data.py
```

Train the RUL model:

```bash
python src/train_rul.py
```

Train the failure prediction model:

```bash
python src/train_failure.py
```

Train and save the machine learning pipelines:

```bash
python src/train_pipeline.py
```

Generate visualizations:

```bash
python src/visualize_results.py
```

## 10. Limitations and Future Improvements

- The current dataset is simulated rather than collected from real machines.
- More realistic sensor data could improve evaluation.
- Additional models could be compared with Logistic Regression and Linear Regression.
- The system could be extended with a streaming platform such as Apache Kafka.
- A dashboard could display sensor readings, predicted failures, and remaining useful life.
- Further testing is required before using this approach in real industrial environments.

## 11. Conclusion

This project demonstrates a basic machine learning pipeline for predicting machine remaining useful life and failure risk from sensor telemetry. It includes data generation, model training, evaluation, visualization, and model saving.
