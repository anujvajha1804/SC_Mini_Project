"""
ann_model.py
Artificial Neural Network (MLPRegressor) training, evaluation, and inference.
"""

from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def train_ann(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = 0.8,
) -> Tuple[MLPRegressor, StandardScaler, StandardScaler, Dict[str, Any]]:
    """
    Train an MLPRegressor using a chronological train/test split.
    
    Parameters:
        X: Feature DataFrame
        y: Target Series (day t+1 solar radiation in W/m²)
        train_ratio: Proportion of data used for training (default 0.8)
        
    Returns:
        (model, scaler_X, scaler_y, results_dict)
    """
    n_samples = len(X)
    if n_samples < 10:
        raise ValueError(f"Too few samples for training ({n_samples}). At least 10 required.")

    split_idx = int(n_samples * train_ratio)
    if split_idx == 0 or split_idx >= n_samples:
        raise ValueError("Invalid split: train or test set would be empty.")

    # Chronological split: first 80% train, last 20% test (NO shuffling to avoid time leakage)
    X_train = X.iloc[:split_idx].copy()
    y_train = y.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_test = y.iloc[split_idx:].copy()

    # StandardScaler for X and y fit ONLY on training set
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

    # MLPRegressor with required hyper-parameters
    model = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        alpha=0.001,
        max_iter=500,
        early_stopping=False,
        random_state=42,
    )

    model.fit(X_train_scaled, y_train_scaled)

    # Predict on test set
    y_pred_scaled = model.predict(X_test_scaled)
    # Inverse scale predictions back to W/m²
    y_pred_test = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    # Ensure non-negative solar radiation predictions
    y_pred_test = np.clip(y_pred_test, 0.0, None)

    results = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred_test": y_pred_test,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    return model, scaler_X, scaler_y, results


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute standard regression performance metrics: MAE, MSE, RMSE, R².
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_t, y_p))
    mse = float(mean_squared_error(y_t, y_p))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_t, y_p))

    return {
        "MAE": round(mae, 3),
        "MSE": round(mse, 3),
        "RMSE": round(rmse, 3),
        "R2": round(r2, 4),
    }


def compute_baseline(X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Compute persistence baseline ('tomorrow = today'):
    Uses today's solar radiation (solar_lag_0) as the forecast for tomorrow (day t+1).
    """
    y_baseline = X_test["solar_lag_0"].values
    metrics = evaluate_metrics(y_test.values, y_baseline)
    return metrics, y_baseline


def predict_future(
    model: MLPRegressor,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
    future_row: pd.DataFrame,
) -> float:
    """
    Predict solar radiation for next day given the latest feature vector.
    """
    future_scaled = scaler_X.transform(future_row)
    pred_scaled = model.predict(future_scaled)
    pred_wm2 = float(scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0])
    # Enforce non-negative physical boundary
    return max(0.0, round(pred_wm2, 2))
