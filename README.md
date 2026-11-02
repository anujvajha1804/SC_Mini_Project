# SolarCast : Solar Index Predictor
> **Next-Day Solar Radiation Prediction via Artificial Neural Network (MLP)**  
> *Academic Mini-Project for Soft Computing Lab (TY B.Tech)*

---

## Overview

**SolarCast** is a lightweight machine learning pipeline that forecasts the next-day **Solar Index** using current-day meteorological observations and past solar history.

* **Solar Index Definition:** Continuous 24-hour average solar radiation in **W/m²** (Watts per square meter).
* **Physical Conversion:**
  ```text
  Solar Index (W/m²) = (Shortwave Radiation in MJ/m² × 1,000,000) / 86,400 seconds
  ```

---

## Tech Stack & File Structure

* **Language:** Python 3.9+
* **Libraries:** `scikit-learn`, `pandas`, `numpy`, `requests`
* **Data Source:** Free Open-Meteo Historical Weather & Geocoding APIs (no API key required)

```text
Mini Project/
├── main.py              # CLI entry point (geocoding, training & prediction)
├── data_collection.py   # Open-Meteo API fetcher & offline fallback logic
├── preprocessing.py     # Data cleaning, outlier checks & 14-feature engineering
├── ann_model.py         # MLPRegressor training, chronological split & evaluation
├── test_suite.py        # Automated test suite
├── requirements.txt     # Dependency list
├── README.md            # Project documentation
└── sample_data/
    └── mumbai_sample.csv # 11-year verified Mumbai dataset (2015–2026)
```

---

## Quickstart

### 1. Installation
```bash
python3 -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the CLI
```bash
# Interactive mode (prompts for city or defaults to Mumbai)
python3 main.py

# Specify any global city directly
python3 main.py --city "Mumbai, Maharashtra"
python3 main.py --city "New Delhi, India"

# Run offline using bundled verified sample data
python3 main.py --offline
```

### 3. Run Automated Tests
```bash
python3 test_suite.py
```

---

## Model & Feature Engineering

### 14 Input Features (Day $t$ to predict Day $t+1$)
1. **Current Weather (5):** Temperature, Relative Humidity, Cloud Cover, Wind Speed, Surface Pressure.
2. **Solar History (7):** Autoregressive daily radiation lags from Day $t$ down to Day $t-6$.
3. **Solar Geometry (2):** Cyclical $\sin$ and $\cos$ of Day-of-Year to model seasonal sun angle continuously.

### Neural Network Architecture
* **Model:** Multilayer Perceptron (`MLPRegressor`) with **1,025 trainable parameters**.
* **Topology:** `14 Inputs -> Dense 32 (ReLU) -> Dense 16 (ReLU) -> Output 1 (Linear)`.
* **Split:** Chronological 80/20 partition (3,452 train, 864 test) with zero look-ahead leakage.
* **Optimization:** Adam optimizer ($\eta = 0.001$), MSE loss, and $L_2$ regularization ($\alpha = 0.001$).
* **Constraint:** Output clipped at >= 0.0 W/m² to enforce physical non-negativity.

