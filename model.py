"""
model.py — CrashSense Real-Time Risk Prediction
Handles model loading and risk prediction logic.
"""

import os
import joblib
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
#  Feature columns expected by the model
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "Signal_Count",
    "Junction_Count",
    "Crossing_Count",
    "Stop_Count",
    "Traffic_Calming_Count",
    "Roundabout_Count",
    "Avg_Visibility",
    "Rain_Accidents",
    "Avg_Hour",
] 

RISK_COLORS = {
    "High":   "#EF4444",   # red
    "Medium": "#F97316",   # orange
    "Low":    "#22C55E",   # green
}


def load_model(model_path: str = None):
    """
    Load the trained RandomForest model from disk.

    Parameters
    ----------
    model_path : str, optional
        Path to the .pkl file. Defaults to models/risk_model_v2.pkl
        relative to this script's directory.

    Returns
    -------
    sklearn model
    """
    if model_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "models", "risk_model_v2.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at: {model_path}\n"
            "Please ensure risk_model_v2.pkl is in the models/ directory."
        )

    model = joblib.load(model_path)
    return model


def predict_real_time_risk(grid: pd.DataFrame, user_inputs: dict, model) -> pd.DataFrame:
    """
    Override selected grid features with real-time / simulated values,
    then run the model to produce updated Risk_Level predictions.

    Parameters
    ----------
    grid : pd.DataFrame
        Pre-processed, gridded accident data with FEATURE_COLS present.
    user_inputs : dict
        Keys can include:
          - 'Avg_Hour'       : float  (0–23)
          - 'Rain_Accidents' : float  (scaled count per grid cell)
          - 'Avg_Visibility' : float  (miles)
    model : sklearn estimator
        Loaded RandomForest model.

    Returns
    -------
    pd.DataFrame
        Copy of grid with updated feature values and new 'Risk_Level' column.
    """
    grid_rt = grid.copy()

    # Apply overrides
    for col, val in user_inputs.items():
        if col in grid_rt.columns:
            grid_rt[col] = val

    # Predict
    X = grid_rt[FEATURE_COLS]
    grid_rt["Risk_Level"] = model.predict(X)

    return grid_rt


def get_feature_importances(model) -> pd.Series:
    """Return a sorted Series of feature importances."""
    return pd.Series(
        model.feature_importances_,
        index=FEATURE_COLS,
    ).sort_values(ascending=False)
