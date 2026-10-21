import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.set_page_config(page_title="MoES SIH26080 Prototype", layout="wide")

st.title(" Aware AI Post-Processing of Monsoon Rainfall Forecasts (SIH26080)")
st.write("Ministry of Earth Sciences (MoES) | Maharashtra Disaster Management Dashboard")

# Step 1: Saved Models Load Karna
try:
    scaler = joblib.load("models/scaler.pkl")
    kmeans = joblib.load("models/kmeans.pkl")
    regime_names = joblib.load("models/regime_names.pkl")
    xgb_model = joblib.load("models/xgb.pkl")
    rf_model = joblib.load("models/rf.pkl")
    cluster_features = joblib.load("models/cluster_features.pkl")
    train_features = joblib.load("models/features.pkl")
    heavy_rain_features = joblib.load("models/heavy_rain_features.pkl")
    eval_metrics = joblib.load("models/eval_metrics.pkl")
    classifier_metrics = joblib.load("models/classifier_metrics.pkl")
    df = pd.read_csv("weather_data.csv")
except Exception as e:
    st.error(f"Error loading models or dataset. Pehle fetch.py aur train2.py chalayein! Details: {e}")
    st.stop()

# Step 2: Sidebar Controls
st.sidebar.header("Forecast & District Controls")
districts = sorted(df["district"].unique())
selected_district = st.sidebar.selectbox("Select District", districts)

# District ka default current data extract karein
dist_row = df[df["district"] == selected_district].iloc[-1]

raw_forecast = st.sidebar.slider("Raw NWP Forecast (mm)", 0.0, 150.0, float(dist_row["raw_forecast"]))
temp = st.sidebar.slider("Temperature (°C)", 15.0, 45.0, float(dist_row["temperature_2m"]))
humidity = st.sidebar.slider("Relative Humidity (%)", 40.0, 100.0, float(dist_row["relative_humidity_2m"]))
dew_point = st.sidebar.slider("Dew Point (°C)", 10.0, 30.0, float(dist_row["dew_point_2m"]))
pressure = st.sidebar.slider("Surface Pressure (hPa)", 950.0, 1025.0, float(dist_row["surface_pressure"]))
wind = st.sidebar.slider("Wind Speed (km/h)", 0.0, 80.0, float(dist_row["wind_speed_10m"]))
wind_dir = st.sidebar.slider("Wind Direction (°)", 0.0, 360.0, float(dist_row["wind_direction_10m"]))
cloud = st.sidebar.slider("Cloud Cover (%)", 0.0, 100.0, float(dist_row["cloud_cover"]))
cape = st.sidebar.slider("CAPE (J/kg)", 0.0, 4000.0, float(dist_row["cape"]))
tcwv = st.sidebar.slider("Total Column Water Vapour (kg/m²)", 10.0, 90.0, float(dist_row["total_column_water_vapour"]))
prev_rain = st.sidebar.slider("Previous Day Rainfall (mm)", 0.0, 150.0, float(dist_row["previous_rainfall"]))
anomaly = st.sidebar.slider("Rainfall Anomaly (mm)", -40.0, 60.0, float(dist_row["rainfall_anomaly"]))

# Step 3: Live Regime Detection (K-Means)
# FIX: this array previously had 9 values (including `cape`, which the
# clustering model was never trained on) in an order that didn't match
# `cluster_features`. StandardScaler/KMeans need the exact same features,
# in the exact same order, as train2.py used. cluster_features is loaded
# from disk specifically so this never drifts out of sync again.
cluster_input_dict = {
    "temperature_2m": temp,
    "relative_humidity_2m": humidity,
    "dew_point_2m": dew_point,
    "surface_pressure": pressure,
    "wind_speed_10m": wind,
    "cloud_cover": cloud,
    "total_column_water_vapour": tcwv,
    "rainfall_anomaly": anomaly
}
cluster_input = pd.DataFrame([cluster_input_dict])[cluster_features]
scaled_input = scaler.transform(cluster_input)
cluster_id = kmeans.predict(scaled_input)[0]
current_regime = regime_names[cluster_id]  # now a string: "Break"/"Active"/"Depression"

# Step 4: AI Bias Correction Calculation (XGBoost)
precip_prob = min(100.0, max(0.0, (humidity - 50) * 0.5 + cloud * 0.3 + min(cape / 50.0, 30)))

input_dict = {
    "raw_forecast": [raw_forecast],
    "temperature_2m": [temp],
    "relative_humidity_2m": [humidity],
    "dew_point_2m": [dew_point],
    "surface_pressure": [pressure],
    "wind_speed_10m": [wind],
    "wind_direction_10m": [wind_dir],
    "cloud_cover": [cloud],
    "precipitation_probability": [precip_prob],
    "cape": [cape],
    "total_column_water_vapour": [tcwv],
    "previous_rainfall": [prev_rain],
    "rainfall_anomaly": [anomaly],
    "regime_Break": [current_regime == "Break"],
    "regime_Active": [current_regime == "Active"],
    "regime_Depression": [current_regime == "Depression"]
}

input_df = pd.DataFrame(input_dict)[train_features]
corrected_rain = max(0.0, round(float(xgb_model.predict(input_df)[0]), 2))

# Step 5: Heavy Rainfall Risk (Random Forest)
heavy_input = input_df[heavy_rain_features]
probs = rf_model.predict_proba(heavy_input)[0]
heavy_prob = round(float(probs[1] * 100), 1) if len(probs) > 1 else 0.0

# Step 6: UI Metrics
st.subheader(" AI Forecast & Regime Output")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Detected Regime", current_regime)
c2.metric("Raw NWP(Numerical Weather Prediction) Forecast", f"{raw_forecast} mm")
c3.metric("AI Corrected Forecast", f"{corrected_rain} mm", delta=round(corrected_rain - raw_forecast, 2))
c4.metric("Heavy Rain Risk (≥64.5mm)", f"{heavy_prob}%")

# Step 7: Disaster Warning Alert
if heavy_prob >= 60:
    st.error(f" RED ALERT: Extreme Heavy Rainfall likely in {selected_district}! High flood & waterlogging risk.")
elif heavy_prob >= 30:
    st.warning(f" YELLOW ALERT: Moderate chance of intense localized showers in {selected_district}.")
else:
    st.success(f" GREEN ALERT: No immediate severe rainfall threat in {selected_district}.")

# Step 8: Spatial Map of Maharashtra
st.subheader(" Regional Spatial Map ")
district_summary = df.groupby("district").agg({
    "latitude": "first",
    "longitude": "first",
    "raw_forecast": "last"
}).reset_index()

# District specific forecast value update
district_summary["corrected_rainfall"] = district_summary["raw_forecast"] * 0.95
district_summary.loc[district_summary["district"] == selected_district, "corrected_rainfall"] = corrected_rain

fig = px.scatter_map(
    district_summary,
    lat="latitude",
    lon="longitude",
    size="corrected_rainfall",
    color="corrected_rainfall",
    hover_name="district",
    zoom=5.5,
    center={"lat": 18.9, "lon": 75.5},
    map_style="carto-positron",
    title="Post-Processed Rainfall Prediction across Maharashtra (mm)"
)
st.plotly_chart(fig, use_container_width=True)

# Step 9: Verification Metrics
st.subheader("Verification Engine (Held-out Test Performance)")
st.dataframe(eval_metrics, use_container_width=True)



st.subheader("Heavy-Rain Classifier Quality (imbalance-aware)")
st.dataframe(classifier_metrics, use_container_width=True)

