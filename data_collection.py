"""
data_collection.py
Module for fetching historical daily weather and solar radiation data from Open-Meteo API.
"""

from datetime import date, datetime, timedelta
from typing import Dict, Optional, Tuple
import pandas as pd
import requests

ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"


def convert_mj_to_wm2(mj_value: float) -> float:
    """
    Convert daily shortwave radiation sum from MJ/m² to average daily W/m².
    Formula: value_MJ * 1,000,000 Joules / 86,400 seconds in a day.
    """
    if pd.isna(mj_value):
        return mj_value
    return float(mj_value) * 1_000_000.0 / 86_400.0


def fetch_open_meteo_daily(
    latitude: float = 19.0760,
    longitude: float = 72.8777,
    start_date: str = "2015-01-01",
    end_date: Optional[str] = None,
    timeout: int = 30,
) -> Tuple[pd.DataFrame, str]:
    """
    Fetch historical daily weather and solar radiation data from Open-Meteo.
    
    Parameters:
        latitude: Latitude of location (-90 to 90)
        longitude: Longitude of location (-180 to 180)
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD). If None, defaults to current date.
        timeout: Request timeout in seconds.
        
    Returns:
        Tuple of (pd.DataFrame, latest_available_date_str)
    """
    # Coordinate validation
    if not (-90.0 <= latitude <= 90.0):
        raise ValueError(f"Invalid latitude: {latitude}. Must be between -90 and 90.")
    if not (-180.0 <= longitude <= 180.0):
        raise ValueError(f"Invalid longitude: {longitude}. Must be between -180 and 180.")

    # Date validation
    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")

    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    except ValueError as e:
        raise ValueError(f"Dates must be in YYYY-MM-DD format: {e}")

    if start_dt > end_dt:
        raise ValueError(f"Start date ({start_date}) cannot be after end date ({end_date}).")

    daily_params = [
        "temperature_2m_mean",
        "relative_humidity_2m_mean",
        "cloud_cover_mean",
        "wind_speed_10m_mean",
        "surface_pressure_mean",
        "shortwave_radiation_sum",
    ]

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(daily_params),
        "timezone": "auto",
    }

    try:
        response = requests.get(ARCHIVE_API_URL, params=params, timeout=timeout)
    except requests.exceptions.Timeout:
        raise ConnectionError(
            "Connection to Open-Meteo API timed out. Please check your network connection or upload a CSV file instead."
        )
    except requests.exceptions.RequestException as e:
        raise ConnectionError(
            f"Network error while connecting to Open-Meteo API ({e}). Please upload a CSV file instead."
        )

    if response.status_code != 200:
        try:
            err_json = response.json()
            reason = err_json.get("reason", response.text)
        except Exception:
            reason = response.text
        raise RuntimeError(
            f"Open-Meteo API returned error {response.status_code}: {reason}. If the API is unavailable, please upload a CSV file."
        )

    try:
        data = response.json()
    except Exception as e:
        raise ValueError(f"Failed to parse API response as JSON: {e}")

    daily = data.get("daily", {})
    if not daily or "time" not in daily or len(daily["time"]) == 0:
        raise ValueError("Open-Meteo API returned an empty dataset for the specified coordinates and date range.")

    df = pd.DataFrame(
        {
            "date": daily.get("time", []),
            "temperature": daily.get("temperature_2m_mean", []),
            "humidity": daily.get("relative_humidity_2m_mean", []),
            "cloud_cover": daily.get("cloud_cover_mean", []),
            "wind_speed": daily.get("wind_speed_10m_mean", []),
            "pressure": daily.get("surface_pressure_mean", []),
            "solar_radiation_mj": daily.get("shortwave_radiation_sum", []),
        }
    )

    if df.empty:
        raise ValueError("No data returned by Open-Meteo API for the selected range.")

    # Fallback: if essential variables are missing from daily response, aggregate hourly data
    missing_cols = [c for c in ["temperature", "humidity", "cloud_cover", "wind_speed", "pressure"] if df[c].isna().all()]
    if missing_cols or df["solar_radiation_mj"].isna().all():
        df = _fetch_and_aggregate_hourly(latitude, longitude, start_date, end_date, timeout)

    # Convert radiation sum (MJ/m²) to average daily irradiance (W/m²)
    df["solar_radiation"] = df["solar_radiation_mj"].apply(convert_mj_to_wm2)
    df.drop(columns=["solar_radiation_mj"], inplace=True, errors="ignore")

    # Filter out trailing dates without recorded solar values
    df = df.dropna(subset=["date"])
    valid_mask = df["solar_radiation"].notna()
    if not valid_mask.any():
        raise ValueError("API returned no valid solar radiation values for the specified range.")

    last_valid_idx = df[valid_mask].index[-1]
    df = df.loc[:last_valid_idx].copy()
    latest_available_date = str(df.loc[last_valid_idx, "date"])

    return df, latest_available_date


def _fetch_and_aggregate_hourly(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    timeout: int,
) -> pd.DataFrame:
    """
    Fallback method: Fetch hourly variables and average them per day.
    """
    hourly_vars = [
        "temperature_2m",
        "relative_humidity_2m",
        "cloud_cover",
        "wind_speed_10m",
        "surface_pressure",
        "shortwave_radiation",
    ]
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(hourly_vars),
        "timezone": "auto",
    }
    resp = requests.get(ARCHIVE_API_URL, params=params, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"Hourly fallback API failed with code {resp.status_code}: {resp.text}")

    h_data = resp.json().get("hourly", {})
    hdf = pd.DataFrame(
        {
            "time": pd.to_datetime(h_data.get("time", [])),
            "temperature": h_data.get("temperature_2m", []),
            "humidity": h_data.get("relative_humidity_2m", []),
            "cloud_cover": h_data.get("cloud_cover", []),
            "wind_speed": h_data.get("wind_speed_10m", []),
            "pressure": h_data.get("surface_pressure", []),
            "solar_hourly_wm2": h_data.get("shortwave_radiation", []),
        }
    )
    hdf["date"] = hdf["time"].dt.date.astype(str)
    # Aggregate daily means
    daily_agg = hdf.groupby("date").agg(
        {
            "temperature": "mean",
            "humidity": "mean",
            "cloud_cover": "mean",
            "wind_speed": "mean",
            "pressure": "mean",
            "solar_hourly_wm2": "sum",  # sum of W/m² hourly gives W*h/m² -> convert to MJ/m²
        }
    ).reset_index()
    # 1 W*h/m² = 3600 J/m² = 0.0036 MJ/m²
    daily_agg["solar_radiation_mj"] = daily_agg["solar_hourly_wm2"] * 3600.0 / 1_000_000.0
    daily_agg.drop(columns=["solar_hourly_wm2"], inplace=True)
    return daily_agg
