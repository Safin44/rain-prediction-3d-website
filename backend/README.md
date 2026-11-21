# ATMOS — Rain Prediction Backend

FastAPI backend that loads the real model from
`random_forest_rain_model.pkl` and serves live predictions to the ATMOS
frontend. The browser cannot load a `.pkl`, so predictions must go
through this backend — there is no in-browser / JavaScript-rule fallback.

## What the .pkl contains (inspected, user-approved load)
- `RandomForestClassifier`, `n_estimators=200`, `classes_=[0, 1]`
- `n_features_in_=8`, `feature_names_in_` in notebook order:
  MinTemp, MaxTemp, Rainfall, Humidity, Pressure, WindSpeed,
  WindGustSpeed, Cloud
- **No scaler inside the .pkl.** The notebook scaled all 8 features with
  a `StandardScaler` fitted on the train split only
  (`train_test_split(test_size=0.2, random_state=42, stratify=y)`).
  That scaler is rebuilt exactly and stored in
  `preprocessing_artifacts.json` (see `verify_model.py`).

## Verify (reproduces the notebook exactly)
```powershell
cd backend
python verify_model.py
```
Expected: accuracy 1.0000, ROC-AUC 1.0000, confusion
`[[9116, 0], [0, 2084]]`, first test row prediction 0 (actual 0).
Unscaled inputs do **not** work (29.48% accuracy) — scaling is required.

## Run the backend (port 8001 — the churn backend uses 8000)
```powershell
cd backend
python -m pip install -r requirements.txt   # pins scikit-learn==1.9.1, the version the .pkl was saved with
python -m uvicorn app:app --host 127.0.0.1 --port 8001
```

## Run the frontend
With the backend running, open `~/workspace/rain_3d_website/index.html`
in a browser (double-click works — the page calls
`http://127.0.0.1:8001/predict`). Internet is needed once for the
Three.js CDN. If the backend is not running, the Weather Station shows
a clear “Backend not running” error instead of guessing.

## Endpoints
- `GET /health` — model/scaler status, model type, tree count, scikit-learn version, feature ranges
- `GET /metrics` — verified notebook metrics (all 5 models with train/test accuracy, AUC and
  confusion matrices; RF precision/recall/F1; importances; tree thresholds; dataset stats)
- `POST /predict` — body: all 8 features
  (`MinTemp, MaxTemp, Rainfall, Humidity, Pressure, WindSpeed,
  WindGustSpeed, Cloud`), each within the dataset range (else HTTP 422);
  returns `prediction` (0/1), `label` (`RAIN` / `NO RAIN`),
  `rain_probability`, `rain_probability_percent`, `trees_voting_rain`,
  `n_estimators`, `threshold_distance` and `model`.
