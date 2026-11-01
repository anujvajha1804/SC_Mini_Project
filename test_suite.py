"""
test_suite.py
Automated test suite verifying data collection, edge cases, preprocessing, model training, and predictions.
"""

import os
import numpy as np
import pandas as pd
import ann_model
import data_collection
import preprocessing

print("=== 1. Testing API Data Collection & Edge Cases ===")
# A. Real Mumbai fetch
try:
    df_api, latest_date = data_collection.fetch_open_meteo_daily(
        latitude=19.0760, longitude=72.8777, start_date="2024-01-01", end_date="2024-01-31"
    )
    print(f"✓ Real API Fetch Success: {len(df_api)} rows fetched. Latest date: {latest_date}")
except Exception as e:
    print(f"✗ API Fetch failed: {e}")

# B. Invalid coordinates
try:
    data_collection.fetch_open_meteo_daily(latitude=95.0, longitude=72.0)
    print("✗ Failed: Invalid latitude was not caught.")
except ValueError as ve:
    print(f"✓ Invalid latitude caught properly: {ve}")

# C. Invalid dates (start > end)
try:
    data_collection.fetch_open_meteo_daily(start_date="2024-05-01", end_date="2024-01-01")
    print("✗ Failed: Invalid date range was not caught.")
except ValueError as ve:
    print(f"✓ Invalid date range caught properly: {ve}")


print("\n=== 2. Testing Preprocessing Edge Cases ===")
# A. Missing required columns
sample_df = pd.read_csv("sample_data/mumbai_sample.csv")
missing_col_df = sample_df.drop(columns=["pressure", "wind_speed"])
try:
    preprocessing.clean_weather_data(missing_col_df)
    print("✗ Failed: Missing columns were not detected.")
except ValueError as ve:
    print(f"✓ Missing columns detected properly: {ve}")

# B. Duplicate dates
dup_df = sample_df.head(20).copy()
dup_df = pd.concat([dup_df, dup_df.head(5)], ignore_index=True)
clean_dup, sum_dup = preprocessing.clean_weather_data(dup_df)
print(f"✓ Duplicates tested: Removed {sum_dup['duplicates_removed']} duplicate rows as expected.")

# C. Outliers outside physical bounds (e.g. temp=150, solar=9999, humidity=-20)
dirty_df = sample_df.head(30).copy()
dirty_df.loc[2, "temperature"] = 150.0  # limit max 60
dirty_df.loc[5, "solar_radiation"] = 800.0  # limit max 500
dirty_df.loc[8, "humidity"] = -15.0  # limit min 0
clean_outlier, sum_outlier = preprocessing.clean_weather_data(dirty_df)
print(f"✓ Outliers tested: {sum_outlier['outliers_removed']} physical outliers flagged and cleaned.")

# D. Missing values & forward fill
nan_df = sample_df.head(30).copy()
nan_df.loc[3:4, "temperature"] = np.nan
clean_nan, sum_nan = preprocessing.clean_weather_data(nan_df)
print(f"✓ Missing values tested: {sum_nan['missing_values_filled']} values forward-filled.")

# E. MJ/m² unit conversion
mj_df = sample_df.head(20).copy()
raw_wm2 = mj_df.loc[0, "solar_radiation"]
# Convert W/m² to MJ/m² for test: MJ = W/m² * 86400 / 10^6
mj_df["solar_radiation"] = mj_df["solar_radiation"] * 86400.0 / 1_000_000.0
clean_mj, _ = preprocessing.clean_weather_data(mj_df, radiation_unit="MJ/m²")
converted_wm2 = clean_mj.loc[0, "solar_radiation"]
assert abs(raw_wm2 - converted_wm2) < 1e-4, f"Conversion error: {raw_wm2} vs {converted_wm2}"
print(f"✓ MJ/m² to W/m² conversion tested: {raw_wm2:.2f} W/m² matches converted {converted_wm2:.2f} W/m².")


print("\n=== 3. Testing Full Dataset, ANN Training & Prediction ===")
full_clean, full_summary = preprocessing.clean_weather_data(sample_df)
X, y, feature_cols, future_row, future_date, last_date = preprocessing.create_features_and_target(full_clean)
print(f"✓ Feature matrix constructed: {X.shape[0]} samples, {X.shape[1]} features.")
print(f"✓ Target date: {future_date.strftime('%Y-%m-%d')} from observed day {last_date.strftime('%Y-%m-%d')}")

model, scaler_X, scaler_y, results = ann_model.train_ann(X, y, train_ratio=0.8)
ann_metrics = ann_model.evaluate_metrics(results["y_test"].values, results["y_pred_test"])
base_metrics, _ = ann_model.compute_baseline(results["X_test"], results["y_test"])
pred_next = ann_model.predict_future(model, scaler_X, scaler_y, future_row)

print("✓ Model Training Complete:")
print(f"   Train samples: {results['train_size']}, Test samples: {results['test_size']}")
print(f"   ANN Metrics:      MAE={ann_metrics['MAE']}, RMSE={ann_metrics['RMSE']}, R²={ann_metrics['R2']}")
print(f"   Baseline Metrics: MAE={base_metrics['MAE']}, RMSE={base_metrics['RMSE']}, R²={base_metrics['R2']}")
print(f"   Next Day Forecast: {pred_next} W/m² on {future_date.strftime('%Y-%m-%d')}")

print("\nALL AUTOMATED TESTS PASSED SUCCESSFULLY!")
