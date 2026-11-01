"""
preprocessing.py
Data cleaning, validation, and feature engineering for solar radiation prediction.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Physical plausible limits for meteorological variables
PHYSICAL_LIMITS = {
    "temperature": (-50.0, 60.0),    # °C
    "humidity": (0.0, 100.0),        # %
    "cloud_cover": (0.0, 100.0),     # %
    "wind_speed": (0.0, 150.0),      # km/h
    "pressure": (800.0, 1100.0),     # hPa
    "solar_radiation": (0.0, 500.0), # W/m²
}

REQUIRED_CANONICAL_COLUMNS = [
    "date",
    "temperature",
    "humidity",
    "cloud_cover",
    "wind_speed",
    "pressure",
    "solar_radiation",
]

# Alias map to handle diverse CSV column header names
COLUMN_ALIASES: Dict[str, List[str]] = {
    "date": ["date", "time", "datetime", "timestamp", "day", "record_date"],
    "temperature": ["temperature", "temp", "temperature_2m", "temperature_2m_mean", "mean_temp", "avg_temp", "t2m"],
    "humidity": ["humidity", "rh", "relative_humidity", "relative_humidity_2m", "relative_humidity_2m_mean", "rh_mean"],
    "cloud_cover": ["cloud_cover", "cloud", "cloudcover", "cloud_cover_mean", "total_cloud_cover", "clouds"],
    "wind_speed": ["wind_speed", "wind", "windspeed", "wind_speed_10m", "wind_speed_10m_mean", "ws"],
    "pressure": ["pressure", "surface_pressure", "surface_pressure_mean", "pres", "barometer", "sp"],
    "solar_radiation": [
        "solar_radiation",
        "solar",
        "radiation",
        "shortwave_radiation",
        "shortwave_radiation_sum",
        "solar_index",
        "solar_irradiance",
        "radiation_wm2",
        "ghi",
    ],
}


def map_columns_to_canonical(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Standardize column names using case-insensitive alias mapping.
    Returns (renamed_df, list_of_missing_required_columns).
    """
    col_mapping = {}
    normalized_cols = {c.strip().lower().replace(" ", "_"): c for c in df.columns}

    for canonical_name, aliases in COLUMN_ALIASES.items():
        matched = None
        for alias in aliases:
            cleaned_alias = alias.lower()
            if cleaned_alias in normalized_cols:
                matched = normalized_cols[cleaned_alias]
                break
        if matched:
            col_mapping[matched] = canonical_name

    renamed_df = df.rename(columns=col_mapping)
    missing_cols = [col for col in REQUIRED_CANONICAL_COLUMNS if col not in renamed_df.columns]
    return renamed_df, missing_cols


def clean_weather_data(
    df_raw: pd.DataFrame,
    radiation_unit: str = "W/m²",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Standardizes, validates, and cleans weather data according to physical constraints.
    
    Order of cleaning:
    1. Standardize column names. Parse dates. Sort by date.
    2. Drop duplicate dates.
    3. Convert all columns to numeric. Bad values become NaN.
    4. Treat values outside physical limits as invalid (NaN).
    5. Fill missing values using forward fill with limit of 3 days. Drop rows still missing.
    6. Return clean data plus summary dictionary.
    """
    summary: Dict[str, Any] = {}
    rows_before = len(df_raw)
    summary["rows_before"] = rows_before

    # 1. Standardize column names
    df_standard, missing = map_columns_to_canonical(df_raw.copy())
    if missing:
        raise ValueError(
            f"Missing required columns in dataset: {', '.join(missing)}. "
            f"Required columns (or aliases): {', '.join(REQUIRED_CANONICAL_COLUMNS)}."
        )

    # Unit conversion if user provided MJ/m²
    if radiation_unit.strip().lower() in ["mj/m²", "mj/m2", "mj"]:
        df_standard["solar_radiation"] = pd.to_numeric(
            df_standard["solar_radiation"], errors="coerce"
        ) * 1_000_000.0 / 86_400.0

    # Parse and sort dates
    df_standard["date"] = pd.to_datetime(df_standard["date"], errors="coerce")
    df_standard = df_standard.dropna(subset=["date"])
    df_standard["date"] = df_standard["date"].dt.normalize()
    df_standard = df_standard.sort_values(by="date").reset_index(drop=True)

    # 2. Drop duplicate dates
    len_before_dup = len(df_standard)
    df_standard = df_standard.drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)
    duplicates_removed = len_before_dup - len(df_standard)
    summary["duplicates_removed"] = duplicates_removed

    # 3. Numeric type conversion
    numeric_cols = ["temperature", "humidity", "cloud_cover", "wind_speed", "pressure", "solar_radiation"]
    for col in numeric_cols:
        df_standard[col] = pd.to_numeric(df_standard[col], errors="coerce")

    nan_count_initial = int(df_standard[numeric_cols].isna().sum().sum())

    # 4. Filter values outside physical limits
    outliers_count = 0
    for col, (min_val, max_val) in PHYSICAL_LIMITS.items():
        invalid_mask = (df_standard[col] < min_val) | (df_standard[col] > max_val)
        outliers_in_col = int(invalid_mask.sum())
        outliers_count += outliers_in_col
        df_standard.loc[invalid_mask, col] = np.nan

    summary["outliers_removed"] = outliers_count
    missing_values_found = nan_count_initial + outliers_count
    summary["missing_values_found"] = missing_values_found

    # 5. Forward-fill missing values (max 3 consecutive days)
    nan_before_ffill = int(df_standard[numeric_cols].isna().sum().sum())
    df_standard[numeric_cols] = df_standard[numeric_cols].ffill(limit=3)
    nan_after_ffill = int(df_standard[numeric_cols].isna().sum().sum())
    missing_values_filled = nan_before_ffill - nan_after_ffill
    summary["missing_values_filled"] = missing_values_filled

    # Drop any remaining unfillable rows
    len_before_drop = len(df_standard)
    df_clean = df_standard.dropna(subset=numeric_cols).reset_index(drop=True)
    rows_dropped = len_before_drop - len(df_clean)
    summary["rows_dropped"] = rows_dropped

    rows_after = len(df_clean)
    summary["rows_after"] = rows_after
    summary["final_columns"] = df_clean.columns.tolist()
    summary["dtypes"] = {col: str(dtype) for col, dtype in df_clean.dtypes.items()}

    return df_clean, summary


def create_features_and_target(
    clean_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, List[str], pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """
    Construct features and target for predicting day t+1 using day t information.
    
    Features for each day t:
    - Day t mean weather: temperature, humidity, cloud_cover, wind_speed, pressure
    - Solar radiation lags: day t (lag 0), t-1, t-2, t-3, t-4, t-5, t-6 (7 values)
    - Target day (t+1) day-of-year sin & cos (cyclical calendar encoding)
    
    Target:
    - Solar radiation at day t+1
    
    Also generates:
    - future_features: The 1-row feature vector built from the very last day of data
      to predict day (last_date + 1).
    - future_pred_date: The date corresponding to last_date + 1 day.
    - last_available_date: The date of the last row in clean_df.
    """
    if len(clean_df) < 14:
        raise ValueError(
            f"Dataset has only {len(clean_df)} clean rows. At least 14 rows are required for 7-day lag features."
        )

    df = clean_df.sort_values(by="date").reset_index(drop=True)
    n = len(df)

    rows = []
    # Build historical dataset: each i represents day t (from index 6 to n-2)
    # Target is day t+1 (index i+1)
    for i in range(6, n - 1):
        target_date = df.loc[i + 1, "date"]
        # Cyclical day of year of TARGET day (t+1)
        doy = target_date.dayofyear
        sin_doy = np.sin(2.0 * np.pi * doy / 365.25)
        cos_doy = np.cos(2.0 * np.pi * doy / 365.25)

        row = {
            "date_t": df.loc[i, "date"],
            "target_date": target_date,
            "temperature_t": df.loc[i, "temperature"],
            "humidity_t": df.loc[i, "humidity"],
            "cloud_cover_t": df.loc[i, "cloud_cover"],
            "wind_speed_t": df.loc[i, "wind_speed"],
            "pressure_t": df.loc[i, "pressure"],
            # 7 daily solar lags: lag 0 (day t) down to lag 6 (day t-6)
            "solar_lag_0": df.loc[i, "solar_radiation"],
            "solar_lag_1": df.loc[i - 1, "solar_radiation"],
            "solar_lag_2": df.loc[i - 2, "solar_radiation"],
            "solar_lag_3": df.loc[i - 3, "solar_radiation"],
            "solar_lag_4": df.loc[i - 4, "solar_radiation"],
            "solar_lag_5": df.loc[i - 5, "solar_radiation"],
            "solar_lag_6": df.loc[i - 6, "solar_radiation"],
            "target_sin_doy": sin_doy,
            "target_cos_doy": cos_doy,
            "target_solar": df.loc[i + 1, "solar_radiation"],
        }
        rows.append(row)

    feature_df = pd.DataFrame(rows)

    feature_cols = [
        "temperature_t",
        "humidity_t",
        "cloud_cover_t",
        "wind_speed_t",
        "pressure_t",
        "solar_lag_0",
        "solar_lag_1",
        "solar_lag_2",
        "solar_lag_3",
        "solar_lag_4",
        "solar_lag_5",
        "solar_lag_6",
        "target_sin_doy",
        "target_cos_doy",
    ]

    X = feature_df[feature_cols].copy()
    y = feature_df["target_solar"].copy()

    # Build future feature row from the very last available row (index n - 1)
    last_idx = n - 1
    last_available_date = df.loc[last_idx, "date"]
    future_pred_date = last_available_date + pd.Timedelta(days=1)
    future_doy = future_pred_date.dayofyear
    future_sin_doy = np.sin(2.0 * np.pi * future_doy / 365.25)
    future_cos_doy = np.cos(2.0 * np.pi * future_doy / 365.25)

    future_row = pd.DataFrame(
        [
            {
                "temperature_t": df.loc[last_idx, "temperature"],
                "humidity_t": df.loc[last_idx, "humidity"],
                "cloud_cover_t": df.loc[last_idx, "cloud_cover"],
                "wind_speed_t": df.loc[last_idx, "wind_speed"],
                "pressure_t": df.loc[last_idx, "pressure"],
                "solar_lag_0": df.loc[last_idx, "solar_radiation"],
                "solar_lag_1": df.loc[last_idx - 1, "solar_radiation"],
                "solar_lag_2": df.loc[last_idx - 2, "solar_radiation"],
                "solar_lag_3": df.loc[last_idx - 3, "solar_radiation"],
                "solar_lag_4": df.loc[last_idx - 4, "solar_radiation"],
                "solar_lag_5": df.loc[last_idx - 5, "solar_radiation"],
                "solar_lag_6": df.loc[last_idx - 6, "solar_radiation"],
                "target_sin_doy": future_sin_doy,
                "target_cos_doy": future_cos_doy,
            }
        ]
    )[feature_cols]

    return X, y, feature_cols, future_row, future_pred_date, last_available_date
