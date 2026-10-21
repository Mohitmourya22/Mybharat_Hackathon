# DisasterReady CG: Community Emergency Response Platform

Smart India Hackathon 2026 | Problem Statement SIH26080 | Team: Part Time Engineers

Chhattisgarh-focused flood and heavy-rain awareness prototype in Chhattisgarhi, Hindi and English.

## What is in this repo

| Path | What it is |
|---|---|
| `docs/index.html` | Public web prototype (live rain forecast, alerts, contacts, report-verification demo). Deploy this folder. |
| `docs/cg-map.png` | Chhattisgarh district base map (source: d-maps.com) |
| `fetch.py` | Downloads ERA5 daily weather for 10 Chhattisgarh districts, monsoon 2020-2025 (Open-Meteo archive API) |
| `train2.py` | Trains regime clustering (KMeans), rainfall bias correction (XGBoost) and heavy-rain classifier (Random Forest) |
| `app.py` | Streamlit dashboard for the ML engine |
| `weather_data.csv`, `models/` | Generated dataset and trained models |

## Run the ML part
```
pip install -r requirements.txt
python fetch.py
python train2.py
streamlit run app.py
```

## Web prototype
Open `docs/index.html` in a browser (needs internet for the live forecast). Deploy the `docs` folder on Netlify or GitHub Pages.

## Data sources
- Live forecast and river flow: Open-Meteo Forecast and Flood APIs
- Training data: ERA5 reanalysis via Open-Meteo archive API
- Rain categories: IMD (Watch >=15.6, Orange >=64.5, Red >=115.6 mm/day)
- Helplines: 112, 108, 100, 1098, 1091, 1077, 1070 and district numbers listed on district websites (checked 22 Oct 2026, call to confirm)

## Honest limitations
- Website alerts are rule-based from forecast rainfall; they are NOT official IMD/SDMA warnings.
- `raw_forecast` in the training data is simulated (actual x 0.8 + noise), not real NWP output. Real historical NWP forecasts are the next step.
- Heavy-rain events are rare (about 1%); the test set has only 12 events, so metrics are indicative.
- Reports in the web demo live in browser memory only. Shelters and a real backend are not included yet.
- Chhattisgarhi text needs native-speaker review.
