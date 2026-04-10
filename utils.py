"""
utils.py — CrashSense Data Preprocessing & Grid Creation
Mirrors the logic from the original CrashSense_Accident_Hotspot_Analysis notebook.
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
#  Preprocessing
# ─────────────────────────────────────────────

def preprocess_data_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and engineer features from the raw US Accidents CSV.

    Steps
    -----
    1. Drop rows missing key columns.
    2. Parse Start_Time → Hour, Is_Night.
    3. Fill bool infrastructure columns.
    4. Fill missing Visibility with median.
    5. Create binary Is_Rain flag.

    Returns
    -------
    pd.DataFrame  (enriched)
    """
    df = df.dropna(subset=["Start_Lat", "Start_Lng", "Start_Time", "Severity"])

    df["Start_Time"] = pd.to_datetime(df["Start_Time"])

    # Time features
    df["Hour"]     = df["Start_Time"].dt.hour
    df["Is_Night"] = (df["Sunrise_Sunset"] == "Night").astype(int)

    # Infrastructure boolean → int
    bool_cols = [
        "Traffic_Signal", "Junction", "Crossing",
        "Stop", "Traffic_Calming", "Roundabout",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(int)

    # Visibility
    if "Visibility(mi)" in df.columns:
        df["Visibility(mi)"] = df["Visibility(mi)"].fillna(
            df["Visibility(mi)"].median()
        )

    # Rain indicator
    if "Weather_Condition" in df.columns:
        df["Is_Rain"] = (
            df["Weather_Condition"]
            .fillna("")
            .str.contains("Rain", case=False)
            .astype(int)
        )
    else:
        df["Is_Rain"] = 0

    return df


# ─────────────────────────────────────────────
#  Spatial Grid Creation
# ─────────────────────────────────────────────

def create_grids(df: pd.DataFrame, grid_size: float = 0.02) -> pd.DataFrame:
    """
    Snap lat/lng to a regular grid of `grid_size` degrees.

    Returns
    -------
    pd.DataFrame  (with Lat_Grid, Lng_Grid columns added)
    """
    df["Lat_Grid"] = (df["Start_Lat"] // grid_size) * grid_size
    df["Lng_Grid"] = (df["Start_Lng"] // grid_size) * grid_size
    return df


def aggregate_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-accident rows into per-grid-cell feature vectors.

    Returns
    -------
    pd.DataFrame  with columns:
        Lat_Grid, Lng_Grid,
        Accident_Count, Avg_Severity, Night_Accidents,
        Signal_Count, Junction_Count, Crossing_Count,
        Stop_Count, Traffic_Calming_Count, Roundabout_Count,
        Avg_Visibility, Rain_Accidents, Avg_Hour
    """
    agg_dict = dict(
        Accident_Count        = ("Severity",         "count"),
        Avg_Severity          = ("Severity",         "mean"),
        Night_Accidents       = ("Is_Night",         "sum"),
        Signal_Count          = ("Traffic_Signal",   "sum"),
        Junction_Count        = ("Junction",         "sum"),
        Crossing_Count        = ("Crossing",         "sum"),
        Stop_Count            = ("Stop",             "sum"),
        Traffic_Calming_Count = ("Traffic_Calming",  "sum"),
        Roundabout_Count      = ("Roundabout",       "sum"),
        Avg_Visibility        = ("Visibility(mi)",   "mean"),
        Rain_Accidents        = ("Is_Rain",          "sum"),
        Avg_Hour              = ("Hour",             "mean"),
    )
    grid = (
        df.groupby(["Lat_Grid", "Lng_Grid"])
          .agg(**agg_dict)
          .reset_index()
    )
    return grid
