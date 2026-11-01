"""
app.py
Streamlit web application for Next-Day Solar Radiation Prediction using an ANN.
Minimalist typography-driven design (no cards, no shadows, no colored boxes).
"""

from datetime import date, datetime, timedelta
import html
import json
import socket
import urllib.error
import urllib.parse
import urllib.request
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

import ann_model
import data_collection
import preprocessing

import streamlit.components.v1 as components

# Page configuration with transparent 1x1 favicon (removes logo from Chrome tab)
st.set_page_config(
    page_title="SolarCast : Solar Index Predictor",
    page_icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=",
    layout="centered",
)

# Remove any default tab favicon links in head
components.html(
    """
    <script>
    function clearTabLogo() {
        try {
            const parentDoc = window.parent.document;
            const icons = parentDoc.querySelectorAll("link[rel*='icon']");
            icons.forEach(el => el.remove());
            const blank = parentDoc.createElement("link");
            blank.rel = "shortcut icon";
            blank.type = "image/png";
            blank.href = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";
            parentDoc.head.appendChild(blank);
        } catch(e) {}
    }
    clearTabLogo();
    setTimeout(clearTabLogo, 200);
    setTimeout(clearTabLogo, 800);
    </script>
    """,
    height=0,
    width=0,
)

# Typography & Minimalist CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700;9..144,800&family=Inter:wght@400;500;600;700&display=swap');

    /* Global Typography & Base Styles */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 18px;
        color: #111827;
        background-color: #FFFFFF !important;
    }

    /* Remove Streamlit Deploy Button & Header Clutter */
    .stAppDeployButton, [data-testid="stAppDeployButton"], header [data-testid="stToolbar"] {
        display: none !important;
    }

    /* Invisible helper components */
    div[data-testid="stCustomComponentV1"] {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Page Container Padding & Centering */
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 680px !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }

    /* Headings (Centered) */
    .main-title {
        font-family: 'Fraunces', serif;
        font-size: 44px;
        font-weight: 700;
        color: #111827;
        line-height: 1.15;
        margin-bottom: 6px;
        text-align: center;
    }

    .sub-title {
        font-family: 'Inter', sans-serif;
        font-size: 18px;
        color: #6B7280;
        margin-bottom: 26px;
        line-height: 1.4;
        text-align: center;
    }

    /* City Search Text Input Full 4-Sided Border Styling */
    div[data-testid="stTextInput"] {
        width: 100% !important;
    }

    div[data-testid="stTextInput"] div[data-baseweb="input"],
    div[data-testid="stTextInput"] [data-testid="stTextInputRootElement"] {
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        background-color: #FFFFFF !important;
        height: 52px !important;
        min-height: 52px !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
        padding: 0 !important;
        box-sizing: border-box !important;
        overflow: hidden !important;
    }

    div[data-testid="stTextInput"] div[data-baseweb="input"]:hover,
    div[data-testid="stTextInput"] [data-testid="stTextInputRootElement"]:hover {
        border-color: #94A3B8 !important;
    }

    div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within,
    div[data-testid="stTextInput"] [data-testid="stTextInputRootElement"]:focus-within {
        border-color: #D9622B !important;
        box-shadow: 0 0 0 1px #D9622B !important;
    }

    div[data-testid="stTextInput"] div[data-baseweb="base-input"] {
        border: none !important;
        background: transparent !important;
        height: 100% !important;
        min-height: 100% !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    div[data-testid="stTextInput"] input {
        border: none !important;
        outline: none !important;
        background: transparent !important;
        font-size: 17px !important;
        color: #111827 !important;
        height: 50px !important;
        line-height: 50px !important;
        padding: 0 14px !important;
        text-align: center !important;
        box-shadow: none !important;
        box-sizing: border-box !important;
    }

    div[data-testid="stTextInput"] input::placeholder {
        color: #9CA3AF !important;
        font-size: 16px !important;
        opacity: 1 !important;
    }

    /* Hide Streamlit's 'Press Enter to apply' instructions completely */
    [data-testid="InputInstructions"],
    div[data-testid="InputInstructions"],
    .stTextInput [data-testid="InputInstructions"] {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Coordinates Caption (Centered, Light Grey Text) */
    .coords-caption {
        font-size: 16px;
        font-weight: 400;
        color: #94A3B8;
        text-align: center;
        margin-top: 8px;
        margin-bottom: 16px;
        letter-spacing: 0.01em;
    }

    /* Compact chart display - remove extra bottom space */
    div[data-testid="stImage"] {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    div[data-testid="stImage"] > img {
        margin-bottom: 0 !important;
    }

    /* Big Full-Width Button */
    div.stButton > button[kind="primary"] {
        height: 56px !important;
        font-size: 20px !important;
        font-weight: 600 !important;
        background-color: #D9622B !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        width: 100% !important;
        margin-top: 10px;
        margin-bottom: 18px;
        transition: opacity 0.15s ease;
    }
    div.stButton > button[kind="primary"]:hover {
        opacity: 0.92;
    }

    /* Result Area (Centered, Whitespace only) */
    .res-date-label {
        font-size: 15px;
        font-weight: 600;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 18px;
        margin-bottom: 2px;
        text-align: center;
    }

    .res-number-row {
        font-family: 'Fraunces', serif;
        font-size: 88px;
        font-weight: 700;
        color: #D9622B;
        line-height: 0.95;
        margin: 8px 0;
        text-align: center;
    }

    .res-unit {
        font-family: 'Inter', sans-serif;
        font-size: 32px;
        font-weight: 500;
        color: #6B7280;
        margin-left: 6px;
    }

    .res-location {
        font-size: 16px;
        color: #6B7280;
        margin-top: 4px;
        margin-bottom: 6px;
        text-align: center;
    }

    .res-meaning {
        font-size: 20px;
        font-weight: 600;
        color: #111827;
        margin-top: 6px;
        margin-bottom: 4px;
        text-align: center;
    }

    .res-caption {
        font-size: 14px;
        color: #9CA3AF;
        margin-top: 4px;
        margin-bottom: 24px;
        text-align: center;
    }

    .fallback-note {
        font-size: 14px;
        color: #6B7280;
        font-style: italic;
        margin-top: 6px;
        margin-bottom: 12px;
        text-align: center;
    }

    /* Metrics Plain Display (Centered) */
    .metric-label {
        font-size: 14px;
        font-weight: 500;
        color: #6B7280;
        margin-bottom: 2px;
        text-align: center;
    }

    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #111827;
        line-height: 1.1;
        text-align: center;
    }

    .metric-summary-line {
        font-size: 16px;
        color: #6B7280;
        margin-top: 12px;
        margin-bottom: 20px;
        text-align: center;
    }

    /* Clean User-Friendly Alert Banners */
    .user-error-banner {
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 14px 0 18px 0;
        color: #991B1B;
        font-size: 15px;
        line-height: 1.5;
        text-align: center;
    }
    .user-error-banner strong {
        font-weight: 600;
        display: block;
        margin-bottom: 3px;
        color: #B91C1C;
    }
    .user-warning-banner {
        background-color: #FFFBEB;
        border: 1px solid #FDE68A;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 14px 0 18px 0;
        color: #92400E;
        font-size: 15px;
        line-height: 1.5;
        text-align: center;
    }
    .user-warning-banner strong {
        font-weight: 600;
        display: block;
        margin-bottom: 3px;
        color: #B45309;
    }
    .user-warning-banner code {
        background-color: #FEF3C7;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 14px;
        color: #78350F;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(show_spinner=False)
def geocode_city_search(query: str):
    """
    Query Open-Meteo Geocoding API for matching coordinates.
    Handles 'City, State' or single city queries.
    Returns dict: {'status': 'ok'|'not_found'|'network_error'|'empty', 'places': list}
    """
    if not query or not query.strip():
        return {"status": "empty", "places": []}
    
    # Try full query first, then fallback to city name if needed
    attempts = [query.strip()]
    if "," in query:
        attempts.append(query.split(",")[0].strip())

    network_error = False
    for q in attempts:
        url = (
            f"https://geocoding-api.open-meteo.com/v1/search?"
            f"name={urllib.parse.quote(q)}&count=5&language=en&format=json"
        )
        try:
            req = urllib.request.urlopen(url, timeout=5)
            data = json.loads(req.read().decode("utf-8"))
            results = data.get("results", [])
            if results:
                places = []
                for r in results:
                    name = r.get("name", "")
                    admin1 = r.get("admin1", "")
                    country = r.get("country", "")
                    parts = [p for p in [name, admin1, country] if p]
                    label = ", ".join(parts)
                    places.append(
                        {
                            "label": label,
                            "latitude": float(r.get("latitude")),
                            "longitude": float(r.get("longitude")),
                        }
                    )
                return {"status": "ok", "places": places}
        except (urllib.error.URLError, TimeoutError, socket.timeout):
            network_error = True
        except Exception:
            pass

    if network_error:
        return {"status": "network_error", "places": []}
    return {"status": "not_found", "places": []}


@st.cache_data(show_spinner=False)
def run_full_prediction_pipeline(lat: float, lon: float, start_date_str: str, end_date_str: str):
    """
    Complete pipeline: fetches, cleans, trains ANN, and returns prediction & metrics.
    """
    raw_df, latest_date = data_collection.fetch_open_meteo_daily(
        latitude=lat,
        longitude=lon,
        start_date=start_date_str,
        end_date=end_date_str,
        timeout=30,
    )

    if len(raw_df) < 60:
        raise ValueError(
            f"This location has only {len(raw_df)} days of data. At least 60 days of historical records are required."
        )

    clean_df, summary = preprocessing.clean_weather_data(raw_df, radiation_unit="W/m²")
    if len(clean_df) < 60:
        raise ValueError(
            f"This location has only {len(clean_df)} clean days. At least 60 days of clean historical records are required."
        )

    X, y, feature_cols, future_row, future_date, last_date = preprocessing.create_features_and_target(clean_df)
    model, scaler_X, scaler_y, results = ann_model.train_ann(X, y, train_ratio=0.8)
    metrics = ann_model.evaluate_metrics(results["y_test"].values, results["y_pred_test"])
    pred_val = ann_model.predict_future(model, scaler_X, scaler_y, future_row)

    test_dates = [d.strftime("%Y-%m-%d") for d in pd.to_datetime(clean_df.loc[results["y_test"].index + 7, "date"])]

    return {
        "pred_val": pred_val,
        "future_date": future_date,
        "last_date_str": latest_date,
        "metrics": metrics,
        "y_test": results["y_test"].values,
        "y_pred": results["y_pred_test"],
        "test_dates": test_dates,
        "clean_df": clean_df,
        "summary": summary,
        "train_size": results["train_size"],
        "test_size": results["test_size"],
        "is_fallback": False,
    }


def run_mumbai_offline_fallback():
    """
    Offline fallback for Mumbai: processes bundled sample data.
    """
    sample_df = pd.read_csv("sample_data/mumbai_sample.csv")
    clean_df, summary = preprocessing.clean_weather_data(sample_df, radiation_unit="W/m²")
    X, y, feature_cols, future_row, future_date, last_date = preprocessing.create_features_and_target(clean_df)
    model, scaler_X, scaler_y, results = ann_model.train_ann(X, y, train_ratio=0.8)
    metrics = ann_model.evaluate_metrics(results["y_test"].values, results["y_pred_test"])
    pred_val = ann_model.predict_future(model, scaler_X, scaler_y, future_row)
    test_dates = [d.strftime("%Y-%m-%d") for d in pd.to_datetime(clean_df.loc[results["y_test"].index + 7, "date"])]

    return {
        "pred_val": pred_val,
        "future_date": future_date,
        "last_date_str": last_date.strftime("%Y-%m-%d"),
        "metrics": metrics,
        "y_test": results["y_test"].values,
        "y_pred": results["y_pred_test"],
        "test_dates": test_dates,
        "clean_df": clean_df,
        "summary": summary,
        "train_size": results["train_size"],
        "test_size": results["test_size"],
        "is_fallback": True,
    }


# Session state for persistent result
if "prediction_result" not in st.session_state:
    st.session_state["prediction_result"] = None

# Vertically center content on the screen before a prediction is made
if st.session_state["prediction_result"] is None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: max(35px, calc(46vh - 220px)) !important;
            transition: padding-top 0.3s ease;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Page Header
st.markdown('<div class="main-title">☀️ SolarCast : Solar Index Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Predict tomorrow\'s average solar radiation (W/m²) for any city.</div>',
    unsafe_allow_html=True,
)

# City Search Input
selected_city = st.text_input(
    "Enter city and state",
    value="",
    placeholder="Enter city and state (e.g. Mumbai, Maharashtra)",
    label_visibility="collapsed",
)

if "prev_city" in st.session_state and st.session_state["prev_city"] != selected_city:
    st.session_state["prediction_result"] = None
st.session_state["prev_city"] = selected_city

chosen_lat = None
chosen_lon = None
chosen_label = None

if selected_city and selected_city.strip():
    geocode_data = geocode_city_search(selected_city)
    if geocode_data["status"] == "ok" and geocode_data["places"]:
        chosen_lat = geocode_data["places"][0]["latitude"]
        chosen_lon = geocode_data["places"][0]["longitude"]
        chosen_label = geocode_data["places"][0]["label"]
        st.markdown(
            f'<div class="coords-caption">Coordinates: {chosen_lat:.4f}° N, {chosen_lon:.4f}° E</div>',
            unsafe_allow_html=True,
        )
    elif geocode_data["status"] == "network_error":
        st.markdown(
            """
            <div class="user-warning-banner">
                <strong>🌐 Connection Issue</strong>
                Unable to reach the location service. Please check your internet connection.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        escaped_city = html.escape(selected_city.strip())
        st.markdown(
            f"""
            <div class="user-warning-banner">
                <strong>⚠️ Location Not Found</strong>
                We couldn't find a match for <em>"{escaped_city}"</em>. Please check the spelling or include the state (e.g. <code>Mumbai, Maharashtra</code> or <code>Patna, Bihar</code>).
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    chosen_lat = None
    chosen_lon = None
    chosen_label = None
    st.session_state["prediction_result"] = None

# Forecast Trigger Button
predict_btn = st.button(
    "Predict Tomorrow's Solar Radiation",
    type="primary",
    use_container_width=True,
)

if predict_btn:
    if not selected_city or not selected_city.strip() or chosen_lat is None:
        st.markdown(
            """
            <div class="user-warning-banner">
                <strong>📍 City Required</strong>
                Please enter a valid city and state before predicting solar radiation.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        with st.spinner("Predicting tomorrow's solar radiation..."):
            start_date_str = "2015-01-01"
            end_date_str = date.today().strftime("%Y-%m-%d")

            try:
                result = run_full_prediction_pipeline(chosen_lat, chosen_lon, start_date_str, end_date_str)
                result["location_label"] = chosen_label
                st.session_state["prediction_result"] = result
            except (ConnectionError, requests.exceptions.RequestException, TimeoutError):
                # Check if city is Mumbai for offline fallback
                if chosen_label and ("mumbai" in chosen_label.lower() or abs(chosen_lat - 19.076) < 0.2):
                    result = run_mumbai_offline_fallback()
                    result["location_label"] = chosen_label
                    st.session_state["prediction_result"] = result
                else:
                    st.markdown(
                        f"""
                        <div class="user-error-banner">
                            <strong>🌐 Weather Service Unavailable</strong>
                            Could not retrieve historical weather records from Open-Meteo for {html.escape(chosen_label or "this location")}. Please check your internet connection and try again.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            except ValueError as ve:
                st.markdown(
                    f"""
                    <div class="user-error-banner">
                        <strong>📉 Data Validation Error</strong>
                        {html.escape(str(ve))}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            except Exception as e:
                # Fallback for Mumbai on other network failures
                if chosen_label and ("mumbai" in chosen_label.lower() or abs(chosen_lat - 19.076) < 0.2):
                    result = run_mumbai_offline_fallback()
                    result["location_label"] = chosen_label or "Mumbai, Maharashtra"
                    st.session_state["prediction_result"] = result
                else:
                    st.markdown(
                        f"""
                        <div class="user-error-banner">
                            <strong>⚙️ Prediction Could Not Be Completed</strong>
                            An unexpected issue occurred while processing data for {html.escape(chosen_label or "this location")}. Please try again or select a nearby major city.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# Results Display
res = st.session_state["prediction_result"]

if res is not None:
    st.markdown("<hr>", unsafe_allow_html=True)

    pred_val = res["pred_val"]
    pred_date_formatted = res["future_date"].strftime("%A, %d %B %Y")
    loc_display = res.get("location_label", chosen_label)

    # Interpreted category
    if pred_val < 120.0:
        meaning = "Low: overcast skies, limited sunshine"
    elif pred_val <= 220.0:
        meaning = "Moderate: partly cloudy, normal sunshine"
    else:
        meaning = "High: clear skies, strong sunshine"

    # Primary forecast presentation
    fut_d = res["future_date"]
    if hasattr(fut_d, "date"):
        fut_d = fut_d.date()

    weekday_date_str = fut_d.strftime("%A, %d %B %Y")
    if fut_d == (date.today() + timedelta(days=1)):
        forecast_heading = f"Forecast for Tomorrow ({weekday_date_str})"
    else:
        forecast_heading = f"Forecast for {weekday_date_str}"

    st.markdown(f'<div class="res-date-label">{forecast_heading}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="res-number-row">{pred_val:.1f}<span class="res-unit">W/m²</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="res-location">{loc_display}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="res-meaning">{meaning}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="res-caption">Solar Index = Average daily solar radiation in W/m²</div>',
        unsafe_allow_html=True,
    )

    # Evaluation Plot: Actual vs Predicted on Test Set
    y_test = res["y_test"]
    y_pred = res["y_pred"]
    test_dates = res.get("test_dates", [])
    n_pts = min(90, len(y_test))

    if test_dates and len(test_dates) >= n_pts:
        plot_dates = [pd.to_datetime(d) for d in test_dates[-n_pts:]]
    else:
        end_dt = pd.to_datetime(res["future_date"]) - pd.Timedelta(days=1)
        plot_dates = pd.date_range(end=end_dt, periods=n_pts, freq="D")

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Inter", "-apple-system", "BlinkMacSystemFont", "sans-serif"]

    fig, ax = plt.subplots(figsize=(8.5, 2.85), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(plot_dates, y_test[-n_pts:], label="Actual Solar Radiation (W/m²)", color="#334155", linewidth=1.5)
    ax.plot(
        plot_dates,
        y_pred[-n_pts:],
        label="ANN Predicted (W/m²)",
        color="#D9622B",
        linewidth=1.5,
        linestyle="--",
    )

    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=7))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(colors="#64748B", labelsize=8.5)
    ax.grid(axis="y", linestyle=":", alpha=0.6, color="#E2E8F0")

    ax.set_title(f"Actual vs. Predicted Solar Radiation (Recent {n_pts} Test Days)", fontsize=11, fontweight="normal", color="#111827", pad=10)
    ax.set_xlabel("Date", fontsize=9, color="#64748B")
    ax.set_ylabel("Solar Radiation (W/m²)", fontsize=9, color="#64748B")
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)

    fig.tight_layout(pad=0.2)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Performance Metrics
    st.markdown("<hr>", unsafe_allow_html=True)
    m = res["metrics"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-label">MAE</div><div class="metric-value">{m["MAE"]:.2f} <span style="font-size:14px;font-weight:400;color:#6B7280;">W/m²</span></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-label">RMSE</div><div class="metric-value">{m["RMSE"]:.2f} <span style="font-size:14px;font-weight:400;color:#6B7280;">W/m²</span></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-label">MSE</div><div class="metric-value">{m["MSE"]:.2f} <span style="font-size:14px;font-weight:400;color:#6B7280;">(W/m²)²</span></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-label">R² Score</div><div class="metric-value">{m["R2"]:.2f}</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="metric-summary-line">On average, the prediction is off by {m["MAE"]:.2f} W/m².</div>',
        unsafe_allow_html=True,
    )

