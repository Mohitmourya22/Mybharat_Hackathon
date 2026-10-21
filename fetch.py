import pandas as pd
import numpy as np
import requests
import time

# Chhattisgarh ke 10 geographical zones ke districts
locations = {
 "Raipur": (21.2514, 81.6296), "Bilaspur": (22.0797, 82.1409),
 "Durg": (21.1904, 81.2849), "Jagdalpur": (19.0704, 82.0080),
 "Raigarh": (21.8974, 83.3950), "Korba": (22.3595, 82.7501),
 "Ambikapur": (23.1186, 83.1977), "Dantewada": (18.9000, 81.3500),
 "Kanker": (20.2700, 81.4900), "Rajnandgaon": (21.0974, 81.0337)
}

# 5 Monsoon Seasons (1 June se 30 September)
monsoon_periods = [
    ("2020-06-01", "2020-09-30"),
    ("2021-06-01", "2021-09-30"),
    ("2022-06-01", "2022-09-30"),
    ("2023-06-01", "2023-09-30"),
    ("2024-06-01", "2024-09-30"),
    ("2025-06-01", "2025-09-30")
]

# NOTE: "cape" was removed from this list on purpose.
# The archive-api.open-meteo.com/v1/archive endpoint serves ERA5 reanalysis,
# which does NOT include CAPE as an hourly output variable. Requesting it
# silently returns null for every hour, which is why 'cape' and (since
# precipitation_probability was derived from cape) 'precipitation_probability'
# were 100% NaN in the dataset. We now compute a bounded convective-instability
# proxy from the surface fields ERA5 actually returns (see below).
hourly_vars = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "cloud_cover",
    "precipitation"
]
hourly_query = ",".join(hourly_vars)


def get_regime(row):
    if row["rainfall_anomaly"] > 15.0 and row["wind_speed_10m"] > 18.0:
        return "Depression"
    elif row["rainfall_anomaly"] < -4.0 or row["relative_humidity_2m"] < 68.0:
        return "Break"
    else:
        return "Active"


all_dfs = []

print("Chhattisgarh Disaster Dataset Fetching Started (Target: 6000+ Rows)...")

for district, (lat, lon) in locations.items():
    print(f"--> Fetching data for: {district}")
    season_frames = []

    for start_date, end_date in monsoon_periods:
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}&"
            f"start_date={start_date}&end_date={end_date}&"
            f"hourly={hourly_query}&"
            f"timezone=Asia%2FKolkata"
        )

        try:
            response = requests.get(url).json()
            if "hourly" not in response:
                continue

            h = response["hourly"]
            df_h = pd.DataFrame({
                "time": pd.to_datetime(h["time"]),
                "temperature_2m": h["temperature_2m"],
                "relative_humidity_2m": h["relative_humidity_2m"],
                "dew_point_2m": h["dew_point_2m"],
                "surface_pressure": h["surface_pressure"],
                "wind_speed_10m": h["wind_speed_10m"],
                "wind_direction_10m": h["wind_direction_10m"],
                "cloud_cover": h["cloud_cover"],
                "precipitation": h["precipitation"]
            })

            # Daily Aggregation
            df_h["date"] = df_h["time"].dt.date
            daily = df_h.groupby("date").agg({
                "temperature_2m": "mean",
                "relative_humidity_2m": "mean",
                "dew_point_2m": "mean",
                "surface_pressure": "mean",
                "wind_speed_10m": "mean",
                "wind_direction_10m": "mean",
                "cloud_cover": "mean",
                "precipitation": "sum"
            }).reset_index()

            daily["latitude"] = lat
            daily["longitude"] = lon
            daily["district"] = district
            daily["actual_rainfall"] = daily["precipitation"].round(2)

            # TCWV approximation (from dew point + pressure, unchanged)
            td = daily["dew_point_2m"]
            vapor_pressure = 6.112 * np.exp((17.67 * td) / (td + 243.5))
            daily["total_column_water_vapour"] = ((vapor_pressure / daily["surface_pressure"]) * 1100).round(2)

            # Raw forecast simulation with systematic bias (unchanged)
            noise = np.random.normal(2.0, 3.5, len(daily))
            daily["raw_forecast"] = np.maximum(0.0, (daily["actual_rainfall"] * 0.80 + noise).round(2))

            # --- CAPE proxy (fixes the all-NaN 'cape' column) ---
            # Built only from surface fields ERA5 archive actually provides.
            # Higher humidity, higher surface temp, and lower pressure all push
            # this up, matching real convective-instability behaviour. Bounded
            # to 0-4000 J/kg to line up with app.py's slider range.
            moisture_term = np.clip(daily["relative_humidity_2m"] - 40, 0, None)
            thermal_term = np.clip(daily["temperature_2m"] - 22, 0, None)
            pressure_term = np.clip(1013.0 - daily["surface_pressure"], 0, None)
            cape_proxy = moisture_term * 25 + thermal_term * 60 + pressure_term * 40
            daily["cape"] = np.clip(cape_proxy, 0, 4000).round(1)

            # Precipitation Probability (now safe: cape is never NaN)
            prob = (
                (daily["relative_humidity_2m"] - 50) * 0.5
                + daily["cloud_cover"] * 0.3
                + np.clip(daily["cape"] / 50.0, 0, 30)
            )
            daily["precipitation_probability"] = np.clip(prob, 0, 100).round(1)

            # Previous rainfall lag (reset at each season boundary - correct,
            # since there's a genuine multi-month gap between monsoon seasons)
            daily["previous_rainfall"] = daily["actual_rainfall"].shift(1).fillna(0.0).round(2)

            # Rounding
            daily["temperature_2m"] = daily["temperature_2m"].round(2)
            daily["relative_humidity_2m"] = daily["relative_humidity_2m"].round(2)
            daily["dew_point_2m"] = daily["dew_point_2m"].round(2)
            daily["surface_pressure"] = daily["surface_pressure"].round(2)
            daily["wind_speed_10m"] = daily["wind_speed_10m"].round(2)
            daily["wind_direction_10m"] = daily["wind_direction_10m"].round(1)
            daily["cloud_cover"] = daily["cloud_cover"].round(1)

            season_frames.append(daily)

        except Exception as e:
            print(f"Error fetching {district} for {start_date}: {e}")

        time.sleep(0.3)

    if not season_frames:
        print(f"WARNING: no data fetched for {district}, skipping.")
        continue

    # --- Fixed anomaly baseline ---
    # Previously, rainfall_anomaly was computed against each SEASON's own
    # mean, which forces every season to average to ~0 anomaly by
    # construction - it can never reflect "wetter/drier than normal" across
    # years. We now compute one climatological mean per district across all
    # 5 monsoon seasons, and use that as the fixed baseline.
    district_df = pd.concat(season_frames, ignore_index=True)
    district_climatology = district_df["actual_rainfall"].mean()
    district_df["rainfall_anomaly"] = (district_df["actual_rainfall"] - district_climatology).round(2)

    # Regime depends on rainfall_anomaly, so it must be computed after the fix above
    district_df["regime"] = district_df.apply(get_regime, axis=1)

    all_dfs.append(district_df)

final_df = pd.concat(all_dfs, ignore_index=True)

ordered_cols = [
    "date", "latitude", "longitude", "district", "raw_forecast",
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "surface_pressure", "wind_speed_10m", "wind_direction_10m",
    "cloud_cover", "precipitation_probability", "cape",
    "total_column_water_vapour", "previous_rainfall", "rainfall_anomaly",
    "regime", "actual_rainfall"
]

final_df = final_df[ordered_cols]
final_df.to_csv("weather_data.csv", index=False)

print("\nSuccess! 'weather_data.csv' created.")
print(f"Total Rows Generated: {len(final_df)} rows across {final_df['district'].nunique()} districts and 5 years!")
print("\nRegime distribution:")
print(final_df["regime"].value_counts())
