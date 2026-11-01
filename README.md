# SolarCast : Solar Index Predictor
### Next-Day Average Solar Radiation Prediction Using an Artificial Neural Network (ANN)

*Academic Mini-Project for Soft Computing Lab (TY B.Tech - Third Year Bachelor of Technology)*

---

## 1. Project Overview & Definition

Solar power generation depends heavily on atmospheric variables like temperature, cloud dynamics, relative humidity, wind velocity, and atmospheric pressure. Because weather shifts continuously, forecasting next-day solar irradiance is essential for electrical grid balancing, battery storage scheduling, and solar farm management.

This project implements an end-to-end, lightweight machine learning system using an **Artificial Neural Network (Multilayer Perceptron Regressor - `MLPRegressor`)** to predict the next day's solar radiation based on meteorological observations available on the current day.

### Mandatory Definition: "Solar Index"
> **Solar Index** in this project is strictly defined as the **average daily solar radiation in W/m²** (Watts per square meter averaged over a complete 24-hour cycle).

In physical weather archives (such as the Open-Meteo Historical API), solar radiation is typically accumulated and reported as daily shortwave radiation sum in **Megajoules per square meter (MJ/m²)**. The system converts this into continuous average power density (W/m²) via the standard physical conversion:

```text
Solar Index (W/m²) = (Shortwave Radiation Sum in MJ/m² × 1,000,000 Joules/MJ) / (86,400 seconds/day)
```

---

## 2. Technology Stack & Design Principles

- **Language:** Python 3.9+
- **Web UI & Dashboard:** Streamlit
- **Machine Learning & Preprocessing:** scikit-learn (`MLPRegressor`, `StandardScaler`, `mean_absolute_error`, `mean_squared_error`, `r2_score`)
- **Data Manipulation:** pandas, numpy
- **Plotting & Visualizations:** matplotlib
- **API Communication:** requests
- **Design Constraint:** Completely free, lightweight, zero external paid services, zero API keys required, no heavy deep learning frameworks (no TensorFlow / PyTorch / Docker).

---

## 3. Project File Structure

```text
Mini Project/
├── app.py                      # Streamlined single-screen web UI with geocoding and 1-click forecast
├── data_collection.py          # Open-Meteo API data fetcher with hourly fallback & units conversion
├── preprocessing.py            # Data cleaning pipeline, alias mapper, outlier filter, lag generator
├── ann_model.py                # MLPRegressor training, chronological split, evaluation & inference
├── test_suite.py               # Automated testing suite for API, cleaning edge cases, and training
├── requirements.txt            # Python dependencies
├── README.md                   # Complete academic documentation and architecture details
└── sample_data/
    └── mumbai_sample.csv       # Real weather dataset fetched from Open-Meteo for Mumbai (2015-2026)
```

---

## 4. Setup and Execution Instructions

### Step 1: Clone or navigate to the directory
```bash
cd "Mini Project"
```

### Step 2: (Optional but recommended) Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the automated test suite
To verify all modules and test edge cases (missing columns, duplicates, physical outliers, API parameters):
```bash
python3 test_suite.py
```

### Step 5: Launch the Streamlit application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 5. How Data is Obtained

### A. Open-Meteo Historical Weather API
- Historical weather data is fetched from the Open-Meteo Historical Weather API (`https://archive-api.open-meteo.com/v1/archive`).
- It is free, public, and requires **no API key**.
- The API query provides daily parameters:
  - `temperature_2m_mean` (°C)
  - `relative_humidity_2m_mean` (%)
  - `cloud_cover_mean` (%)
  - `wind_speed_10m_mean` (km/h)
  - `surface_pressure_mean` (hPa)
  - `shortwave_radiation_sum` (MJ/m²)
- **Delay handling:** The weather archive typically updates up to the current day or previous day (0 to 2 days delay depending on UTC archive processing). The code automatically detects the last date containing non-null solar data to establish day t and forecasts for day t+1.
- **Fallback mechanism:** If the network is unavailable for Mumbai, the application automatically loads the bundled historical dataset (`sample_data/mumbai_sample.csv`) as an offline fallback.

### B. City Geocoding Search
- The app uses the free Open-Meteo Geocoding API (`https://geocoding-api.open-meteo.com/v1/search`) to resolve any city name into latitude and longitude coordinates automatically.
- No manual coordinate entry is needed. Default city: **Mumbai**.

---

## 6. Preprocessing and Data Hygiene

Weather data passes through the single, unified function `clean_weather_data()` in `preprocessing.py`:
1. **Standardize column names:** Map headers to canonical names using the alias dictionary. Parse dates into `datetime64` and sort chronologically.
2. **Drop duplicates:** Identify and remove duplicate date timestamps (`drop_duplicates(subset=['date'])`).
3. **Type conversion:** Cast all meteorological columns to numeric; corrupted entries become `NaN`.
4. **Physical Limits Check:** Detect impossible values outside physical sensor limits and convert them to `NaN`:
   - Temperature: -50 to +60 °C
   - Humidity: 0 to 100 %
   - Cloud Cover: 0 to 100 %
   - Wind Speed: 0 to 150 km/h
   - Pressure: 800 to 1100 hPa
   - Solar Radiation: 0 to 500 W/m²
5. **No-Leakage Imputation:** Missing values are imputed using forward-fill (`ffill`) with a limit of 3 consecutive days. This uses only past observations, preventing future data leakage into historical timestamps. Any remaining `NaN`s are dropped.

---

## 7. Feature Engineering (Day t -> Day t+1)

To forecast solar radiation at day t+1, each sample vector strictly uses information known up to day t:
- **Day t Weather (5 features):** Mean temperature, humidity, cloud cover, wind speed, surface pressure.
- **Autoregressive Solar Lags (7 features):** Solar radiation from day t (lag 0), day t-1, day t-2, day t-3, day t-4, day t-5, day t-6.
- **Cyclical Solar Geometry (2 features):** The sun's angle is cyclical across the 365.25-day astronomical year. To ensure December 31 smoothly wraps into January 1, the target day's (t+1) day-of-year (DOY) is transformed into sine and cosine components:
  `sin(2 * pi * DOY / 365.25), cos(2 * pi * DOY / 365.25)`
- **Target (y):** Solar radiation at day t+1.

---

## 8. Artificial Neural Network Architecture & Training

The model is built using scikit-learn's `MLPRegressor`:

```
Input Features (14)
   │
   ▼
[Dense Layer 1: 32 Neurons, ReLU Activation]
   │
   ▼
[Dense Layer 2: 16 Neurons, ReLU Activation]
   │
   ▼
[Output Layer: 1 Linear Neuron] ──> Day t+1 Solar Index (W/m²)
```

- **Chronological Split:** First 80% of dates for Training, final 20% for Testing. Shuffling is strictly disabled to prevent temporal look-ahead leakage.
- **Feature Scaling:** `StandardScaler` (mean = 0, standard deviation = 1) is fitted strictly on the training set and applied to the test set and inference vectors.
- **Activation Function:** Rectified Linear Unit (ReLU), f(z) = max(0, z), enabling the network to learn non-linear atmospheric relationships.
- **Optimizer:** Adam (Adaptive Moment Estimation) with learning rate adaptation and momentum.
- **Loss Function:** Mean Squared Error (MSE), minimized via backpropagation.
- **Regularization:** L2 regularization penalty (`alpha = 0.001`) helps reduce overfitting.
- **Learning Rule & Parameters:** Learning rule: backpropagation (generalized delta rule) with the Adam optimizer. alpha=0.001 is the L2 penalty, learning rate is 0.001.

---

## 9. Evaluation Metrics & Reading Results

Performance on the unseen 20% test set is evaluated using 4 standard metrics:

1. **Mean Absolute Error (MAE):** Average magnitude of forecast errors in W/m²:
   `MAE = (1/N) * sum(|y_i - y_pred_i|)`
2. **Mean Squared Error (MSE):** Penalizes larger forecasting errors more heavily:
   `MSE = (1/N) * sum((y_i - y_pred_i)²)`
3. **Root Mean Squared Error (RMSE):** The square root of MSE, expressed in the original physical units (W/m²).
4. **Coefficient of Determination (R²):** Proportion of variance in actual solar radiation explained by the ANN model. A score of 1.0 is perfect; 0.0 indicates performance no better than predicting the historical average.

### Persistence Benchmark ("Tomorrow = Today")
In weather forecasting, a standard baseline is the persistence rule: y_pred(t+1) = y(t).
On the test set, the persistence baseline achieves MAE = 22.01 W/m², RMSE = 38.72 W/m², and R² = 0.55. In comparison, the ANN achieves MAE = 23.62 W/m², RMSE = 37.06 W/m², and R² = 0.59. The ANN is better on RMSE and R² but slightly worse on MAE than persistence, and the gain over persistence is small.

Results change slightly with the data end date. Values shown are from the bundled Mumbai sample data.

