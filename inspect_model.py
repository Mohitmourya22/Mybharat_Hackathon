import joblib
import os

if not os.path.isdir("models"):
    raise SystemExit("No 'models/' folder found. Run fetch.py then train2.py first.")

# 1. Feature columns list dekhne ke liye:
features = joblib.load("models/features.pkl")
print("Saved Features:\n", features)

# 2. Regime names mapping dekhne ke liye (cluster_id -> "Break"/"Active"/"Depression"):
regimes = joblib.load("models/regime_names.pkl")
print("\nSaved Regimes:\n", regimes)

# 3. K-Means model ke cluster centers dekhne ke liye:
kmeans = joblib.load("models/kmeans.pkl")
print("\nKMeans Clusters Count:", kmeans.n_clusters)
print("Cluster Centers Shape:", kmeans.cluster_centers_.shape)

# 4. XGBoost model ki important features dekhne ke liye:
xgb_model = joblib.load("models/xgb.pkl")
print("\nXGBoost Feature Importances:")
for f, imp in zip(features, xgb_model.feature_importances_):
    print(f"{f}: {round(float(imp), 4)}")
