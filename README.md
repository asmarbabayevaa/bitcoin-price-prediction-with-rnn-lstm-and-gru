# Bitcoin Price Prediction with RNN (LSTM & GRU)

A deep learning project that predicts Bitcoin prices using Recurrent Neural Networks (LSTM and GRU), with automatic hyperparameter tuning via Optuna.

---

## Overview

This project builds a time series forecasting model to predict Bitcoin's daily opening price. It uses a sliding window approach to create sequences from historical data, then trains either an LSTM or GRU model with the best hyperparameters found through Optuna's optimization framework.

---

## Workflow

```
Load Data → Split Train/Test → Scale → Create Sequences → Tune Hyperparameters → Train Final Model → Evaluate → Predict Future
```

1. **Load & Preprocess** — Load Bitcoin CSV, convert dates, split into train (before 2021) and test (2021)
2. **Scaling** — Normalize prices to [0, 1] using MinMaxScaler
3. **Sequence Creation** — Build 30-day sliding windows for RNN input
4. **Hyperparameter Tuning** — Use Optuna to search best model architecture (20 trials)
5. **Final Model Training** — Train LSTM or GRU with best parameters
6. **Evaluation** — Measure performance with MSE, RMSE, MAE, R²
7. **Future Prediction** — Forecast next 30 days beyond test data

---

## Model Details

The model is dynamically built with the following tunable parameters:

| Parameter | Search Space |
|-----------|-------------|
| RNN Type | LSTM or GRU |
| Number of Layers | 1 to 10 |
| Units per Layer | 20 to 100 |
| Dropout Rate | 0.1 to 0.5 |
| Optimizer | Adam, SGD, RMSprop, Adagrad |
| Learning Rate | 1e-5 to 1e-2 |
| Batch Size | 16 to 64 |

---

## Results

Model performance is evaluated on the 2021 test set using:

- **MSE** — Mean Squared Error
- **RMSE** — Root Mean Squared Error
- **MAE** — Mean Absolute Error
- **R²** — Coefficient of Determination
---

## Notes

- MinMaxScaler is fit only on training data to prevent data leakage
- The test sequence construction includes the last 30 days of training data to ensure the first test prediction has a full lookback window
- Future predictions use a rolling window — each new prediction is fed back as input for the next step
