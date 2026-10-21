import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, accuracy_score, classification_report,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import joblib
import os

os.makedirs("models", exist_ok=True)

df = pd.read_csv("weather_data.csv")
print(f"Step 1 Done: 'weather_data.csv' loaded. Total Rows: {len(df)}")

# ---------------------------------------------------------------------------
# 1. Regime clustering
# ---------------------------------------------------------------------------
# LEAKAGE FIX: 'rainfall_anomaly' is (actual_rainfall - constant), i.e. a
# direct copy of the target. It is REMOVED from every feature list.
# It is still used below, but ONLY to give the clusters their names
# (driest cluster = Break, wettest = Depression), never as a model input.
cluster_features = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "surface_pressure",
    "wind_speed_10m",
    "cloud_cover",
    "total_column_water_vapour"
]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[cluster_features])

weights = compute_sample_weight(class_weight="balanced", y=df["regime"])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(scaled_features, sample_weight=weights)

# Name clusters by mean rainfall anomaly: lowest = Break, highest = Depression.
cluster_means = df.groupby("cluster")["rainfall_anomaly"].mean().sort_values()
order = cluster_means.index.tolist()

regime_names = {
    order[0]: "Break",
    order[1]: "Active",
    order[2]: "Depression"
}
df["predicted_regime"] = df["cluster"].map(regime_names)

regime_mapping = {"Break": 0, "Active": 1, "Depression": 2}
df["regime_encoded"] = df["regime"].map(regime_mapping)
df["predicted_regime_encoded"] = df["predicted_regime"].map(regime_mapping)

accuracy = accuracy_score(df["regime_encoded"], df["predicted_regime_encoded"])
print("Cluster-vs-label Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(df["regime_encoded"], df["predicted_regime_encoded"], zero_division=0))

df["regime_Break"] = df["predicted_regime"] == "Break"
df["regime_Active"] = df["predicted_regime"] == "Active"
df["regime_Depression"] = df["predicted_regime"] == "Depression"

# ---------------------------------------------------------------------------
# 2. Features for bias correction
# ---------------------------------------------------------------------------
# 'rainfall_anomaly' removed (leakage). 'previous_rainfall' is kept: it is
# YESTERDAY's observed rainfall, which is legitimately known at forecast time.
train_features = [
    "raw_forecast",
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "cloud_cover",
    "precipitation_probability",
    "cape",
    "total_column_water_vapour",
    "previous_rainfall",
    "regime_Break",
    "regime_Active",
    "regime_Depression"
]

# Time-aware split PER DISTRICT: first 80% of each district's dates = train,
# last 20% = test. Every district appears in both sets.
df["date"] = pd.to_datetime(df["date"])
train_parts, test_parts = [], []
for _, group in df.sort_values("date").groupby("district"):
    cut = int(len(group) * 0.8)
    train_parts.append(group.iloc[:cut])
    test_parts.append(group.iloc[cut:])
train_df = pd.concat(train_parts).sort_index()
test_df = pd.concat(test_parts).sort_index()

X_train, y_train = train_df[train_features], train_df["actual_rainfall"]
X_test, y_test = test_df[train_features], test_df["actual_rainfall"]

xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
xgb_model.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# 3. Heavy-rain probability model (>= 64.5 mm)
# ---------------------------------------------------------------------------
heavy_rain_features = [col for col in train_features
                       if not col.startswith("regime_") and col != "raw_forecast"]
y_heavy_train = (train_df["actual_rainfall"] >= 64.5).astype(int)
print(f"\nHeavy-rain positive rate - train: {y_heavy_train.mean():.2%}")

rf_model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced_subsample",
    random_state=42
)
rf_model.fit(train_df[heavy_rain_features], y_heavy_train)

# ---------------------------------------------------------------------------
# 4. Verification metrics on the test split
# ---------------------------------------------------------------------------
raw_test = test_df["raw_forecast"].values
ai_test = np.maximum(0.0, xgb_model.predict(X_test))
obs_test = y_test.values
threshold = 64.5


def get_pod_far(obs, pred):
    hits = np.sum((pred >= threshold) & (obs >= threshold))
    fa = np.sum((pred >= threshold) & (obs < threshold))
    miss = np.sum((pred < threshold) & (obs >= threshold))
    pod = hits / (hits + miss) if (hits + miss) > 0 else 0.0
    far = fa / (hits + fa) if (hits + fa) > 0 else 0.0
    return round(float(pod), 2), round(float(far), 2)


r_pod, r_far = get_pod_far(obs_test, raw_test)
a_pod, a_far = get_pod_far(obs_test, ai_test)

eval_metrics = pd.DataFrame({
    "Metric": ["RMSE (mm)", "MAE (mm)", "POD (Heavy Rain Detection)", "FAR (False Alarm Ratio)"],
    "Raw NWP Forecast": [
        round(np.sqrt(mean_squared_error(obs_test, raw_test)), 2),
        round(mean_absolute_error(obs_test, raw_test), 2),
        r_pod,
        r_far
    ],
    "Regime-Aware AI": [
        round(np.sqrt(mean_squared_error(obs_test, ai_test)), 2),
        round(mean_absolute_error(obs_test, ai_test), 2),
        a_pod,
        a_far
    ]
})

y_heavy_test = (test_df["actual_rainfall"] >= 64.5).astype(int)
heavy_test_probs = rf_model.predict_proba(test_df[heavy_rain_features])[:, 1]
heavy_test_preds = (heavy_test_probs >= 0.5).astype(int)

classifier_metrics = pd.DataFrame({
    "Metric": ["Precision", "Recall", "F1 Score", "ROC-AUC"],
    "Heavy-Rain Classifier (RF)": [
        round(precision_score(y_heavy_test, heavy_test_preds, zero_division=0), 2),
        round(recall_score(y_heavy_test, heavy_test_preds, zero_division=0), 2),
        round(f1_score(y_heavy_test, heavy_test_preds, zero_division=0), 2),
        round(roc_auc_score(y_heavy_test, heavy_test_probs), 2) if y_heavy_test.nunique() > 1 else float("nan"),
    ]
})

print("\nVerification metrics:")
print(eval_metrics)
print(f"\nHeavy-rain positive rate - test: {y_heavy_test.mean():.2%}")
print("Heavy-rain events in test set:", int(y_heavy_test.sum()), "of", len(y_heavy_test))
print("\nHeavy-rain classifier metrics:")
print(classifier_metrics)

print("\nMax predicted heavy-rain probability in test:", round(float(heavy_test_probs.max()), 3))
print("\nThreshold sensitivity (test set):")
rows = []
for t in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
    p = (heavy_test_probs >= t).astype(int)
    rows.append({
        "threshold": t,
        "alerts": int(p.sum()),
        "hits": int(((p == 1) & (y_heavy_test.values == 1)).sum()),
        "precision": round(precision_score(y_heavy_test, p, zero_division=0), 2),
        "recall": round(recall_score(y_heavy_test, p, zero_division=0), 2),
    })
print(pd.DataFrame(rows))
# ---------------------------------------------------------------------------
# 5. Save all models and features
# ---------------------------------------------------------------------------
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(kmeans, "models/kmeans.pkl")
joblib.dump(regime_names, "models/regime_names.pkl")
joblib.dump(xgb_model, "models/xgb.pkl")
joblib.dump(rf_model, "models/rf.pkl")
joblib.dump(cluster_features, "models/cluster_features.pkl")
joblib.dump(train_features, "models/features.pkl")
joblib.dump(heavy_rain_features, "models/heavy_rain_features.pkl")
joblib.dump(eval_metrics, "models/eval_metrics.pkl")
joblib.dump(classifier_metrics, "models/classifier_metrics.pkl")

print("\nStep 2 Done: All artifacts and metrics successfully saved in 'models/' folder!")