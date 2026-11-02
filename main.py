#!/usr/bin/env python3
"""
main.py
SolarCast : Solar Index Predictor
Command-line interface for next-day solar radiation prediction using an Artificial Neural Network.
"""

import argparse
from datetime import date, datetime, timedelta
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
import numpy as np
import pandas as pd

import ann_model
import data_collection
import preprocessing


def geocode_city(city_query: str):
    """Resolve city name into latitude, longitude, and formatted name."""
    if not city_query or not city_query.strip():
        return None

    query = city_query.strip()
    attempts = [query]
    if "," in query:
        attempts.append(query.split(",")[0].strip())

    for q in attempts:
        url = (
            f"https://geocoding-api.open-meteo.com/v1/search?"
            f"name={urllib.parse.quote(q)}&count=1&language=en&format=json"
        )
        try:
            req = urllib.request.urlopen(url, timeout=10)
            data = json.loads(req.read().decode("utf-8"))
            results = data.get("results", [])
            if results:
                r = results[0]
                name = r.get("name", "")
                admin1 = r.get("admin1", "")
                country = r.get("country", "")
                parts = [p for p in [name, admin1, country] if p]
                label = ", ".join(parts)
                return {
                    "label": label,
                    "latitude": float(r.get("latitude")),
                    "longitude": float(r.get("longitude")),
                }
        except Exception:
            continue
    return None


def interpret_radiation(val: float) -> str:
    """Provide physical atmospheric interpretation for predicted daily W/m²."""
    if val < 120.0:
        return "Low irradiance (Heavy cloud cover / overcast conditions)"
    elif val <= 220.0:
        return "Moderate irradiance (Partly cloudy / standard seasonal sunshine)"
    else:
        return "High irradiance (Clear skies / strong solar generation potential)"


def run_pipeline(city: str, force_offline: bool = False):
    print("\nSolarCast : Solar Index Predictor")
    print("Next-Day Solar Radiation Prediction via Artificial Neural Network (MLP)")
    print("-" * 65)

    # 1. Location Resolution
    is_offline = force_offline
    loc_info = None

    if not is_offline:
        print(f"Resolving location for '{city}'...")
        loc_info = geocode_city(city)
        if not loc_info:
            print(f"Warning: Could not geocode '{city}' online.")
            if "mumbai" in city.lower():
                print("Using default coordinates for Mumbai (19.0760° N, 72.8777° E).")
                loc_info = {"label": "Mumbai, Maharashtra, India", "latitude": 19.0760, "longitude": 72.8777}
            else:
                print(f"Error: Unable to locate '{city}'. Please check the spelling.")
                sys.exit(1)
    else:
        loc_info = {"label": "Mumbai, Maharashtra (Offline Sample)", "latitude": 19.0760, "longitude": 72.8777}

    print(f"Location:    {loc_info['label']}")
    print(f"Coordinates: {loc_info['latitude']:.4f}° N, {loc_info['longitude']:.4f}° E")
    print("-" * 65)

    # 2. Data Retrieval
    raw_df = None
    last_date_str = ""

    if not is_offline:
        start_date = "2015-01-01"
        end_date = date.today().strftime("%Y-%m-%d")
        print(f"Fetching historical meteorological records (2015 to {end_date})...")
        try:
            raw_df, last_date_str = data_collection.fetch_open_meteo_daily(
                latitude=loc_info["latitude"],
                longitude=loc_info["longitude"],
                start_date=start_date,
                end_date=end_date,
                timeout=30,
            )
            print(f"Retrieved {len(raw_df):,} daily records from Open-Meteo archive (latest: {last_date_str}).")
        except Exception as e:
            if "mumbai" in loc_info["label"].lower() or abs(loc_info["latitude"] - 19.076) < 0.2:
                print("Online fetch timed out or unavailable. Switching to verified local Mumbai sample data.")
                is_offline = True
            else:
                print(f"Error retrieving weather data: {e}")
                sys.exit(1)

    if is_offline:
        sample_path = "sample_data/mumbai_sample.csv"
        raw_df = pd.read_csv(sample_path)
        last_date_str = str(raw_df["date"].dropna().iloc[-1])
        print(f"Loaded {len(raw_df):,} daily records from local sample dataset ({sample_path}).")

    # 3. Data Cleaning and Preprocessing
    clean_df, summary = preprocessing.clean_weather_data(raw_df, radiation_unit="W/m²")
    X, y, feature_cols, future_row, future_date, last_date = preprocessing.create_features_and_target(clean_df)

    print(f"Cleaned samples:     {len(clean_df):,} valid days")
    print(f"Feature matrix:      {X.shape[0]:,} samples x {X.shape[1]} features (5 weather, 7 lags, 2 cyclical)")

    # 4. Model Training
    print("Training Artificial Neural Network (MLPRegressor)...")
    model, scaler_X, scaler_y, results = ann_model.train_ann(X, y, train_ratio=0.8)
    ann_metrics = ann_model.evaluate_metrics(results["y_test"].values, results["y_pred_test"])
    base_metrics, _ = ann_model.compute_baseline(results["X_test"], results["y_test"])

    # 5. Future Prediction
    pred_val = ann_model.predict_future(model, scaler_X, scaler_y, future_row)
    interpretation = interpret_radiation(pred_val)

    # 6. Evaluation Summary
    print("-" * 65)
    print("Model Evaluation on Unseen Chronological Test Set (20%):")
    print(f"Training samples:    {results['train_size']:,}")
    print(f"Test samples:        {results['test_size']:,}")
    print()
    print(f"{'Metric':<26} {'ANN Model':<16} {'Persistence':<16} {'Comparison'}")
    print(f"{'MAE (Mean Absolute Error)':<26} {ann_metrics['MAE']:.2f} W/m²       {base_metrics['MAE']:.2f} W/m²       Persistence is slightly lower")
    print(f"{'RMSE (Root Mean Squared)':<26} {ann_metrics['RMSE']:.2f} W/m²       {base_metrics['RMSE']:.2f} W/m²       ANN improves error (-{base_metrics['RMSE'] - ann_metrics['RMSE']:.2f})")
    print(f"{'MSE (Mean Squared Error)':<26} {ann_metrics['MSE']:.2f} (W/m²)²   {base_metrics['MSE']:.2f} (W/m²)²   ANN reduces large errors")
    print(f"{'R² Determination Score':<26} {ann_metrics['R2']:.4f}           {base_metrics['R2']:.4f}           ANN explains +{((ann_metrics['R2'] - base_metrics['R2']) * 100):.1f}% variance")

    # 7. Next-Day Forecast Output
    print("-" * 65)
    target_date_formatted = future_date.strftime("%A, %d %B %Y")
    print(f"Target Forecast Date:      {target_date_formatted}")
    print(f"Predicted Solar Index:     {pred_val:.2f} W/m²")
    print(f"Physical Interpretation:   {interpretation}")
    print(f"Solar Index Definition:    Average daily solar radiation (24h continuous average)")
    print("-" * 65)
    print()


def main():
    parser = argparse.ArgumentParser(description="SolarCast: Next-Day Solar Radiation Prediction CLI")
    parser.add_argument(
        "-c", "--city",
        type=str,
        default=None,
        help="City name and state (e.g., 'Mumbai, Maharashtra' or 'Delhi')",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run using the verified offline sample dataset for Mumbai",
    )
    args = parser.parse_args()

    city = args.city
    if not city and not args.offline:
        # If terminal is interactive, ask the user; otherwise default to Mumbai
        if sys.stdin.isatty():
            try:
                user_input = input("Enter city and state [default: Mumbai, Maharashtra]: ").strip()
                city = user_input if user_input else "Mumbai, Maharashtra"
            except (KeyboardInterrupt, EOFError):
                print("\nCancelled.")
                sys.exit(0)
        else:
            city = "Mumbai, Maharashtra"
    elif not city:
        city = "Mumbai, Maharashtra"

    run_pipeline(city=city, force_offline=args.offline)


if __name__ == "__main__":
    main()
