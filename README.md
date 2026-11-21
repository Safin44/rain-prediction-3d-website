# ATMOS — Rain Observatory (DS422 Rain Prediction, 3D website)

A deliberately **different** site from the E-Commerce Churn 3D site — nothing is reused:

| | Churn site | This Rain site |
|---|---|---|
| Theme | Dark futuristic dashboard | Light sky / weather-journal |
| Colours | Navy `#070b16`, cyan + violet | Paper `#f3f7f4`, storm teal, sun amber |
| Type | All sans (Segoe UI) | Georgia serif headlines + Verdana body |
| Hero stats | Rectangular glass cards | Circular instrument dials |
| 3D background | Dark particle field | Falling rain streaks + drifting clouds |
| Main 3D | 3D bar chart of accuracy | 3D threshold cube (the rule space) with live point |
| Predictor | FastAPI backend loading `.pkl` (port 8000) | FastAPI backend loading `.pkl` (port 8001) — verdict from `random_forest_rain_model.pkl`, gates shown only as explanation |
| Pipeline | Horizontal pills | Vertical field-notes timeline |
| Results | 3D bars + table | Lollipop "model race" + table |

## Backend is required
The prediction **must** come from the model inside
`random_forest_rain_model.pkl`. A browser cannot load a `.pkl`, so a small
Python (FastAPI) backend in `backend/` loads it once at startup and serves
predictions — the same pattern as the churn site, on port **8001** so the
two backends can run side by side.

The `.pkl` contains **only** the Random Forest (200 trees, 8 features,
`feature_names_in_` = MinTemp, MaxTemp, Rainfall, Humidity, Pressure,
WindSpeed, WindGustSpeed, Cloud) — **no scaler**. The notebook scaled all
8 features with a StandardScaler fitted on the train split only
(80/20 stratified, `random_state=42`). `backend/verify_model.py` rebuilds
that scaler exactly, saves it to `backend/preprocessing_artifacts.json`,
and verifies the model + scaler reproduce the notebook on the full test
set: **100.00% accuracy, ROC-AUC 1.0, confusion [[9116, 0], [0, 2084]]**,
first test row prediction 0 (actual 0). Unscaled inputs fail (29.48%), so
scaling is not optional.

The rule the model rediscovered is still shown on the page as an
explanation, and the gates are displayed — but they do **not** decide the
verdict, and there is no JavaScript-rule fallback. If the backend is not
running, the Weather Station shows a clear “Backend not running” error.

```
RainTomorrow = 1  iff  Humidity > 65  AND  Cloud > 4.5  AND  Rainfall > 2
(verified on 100.00% of all 55,996 rows — explanation only)
```

## Run it (for the viva)
1. Start the backend (Windows PowerShell, from the `rain_3d_website` folder):
```powershell
cd backend
python -m pip install -r requirements.txt   # once — pins scikit-learn 1.9.1 (the .pkl's version)
python -m uvicorn app:app --host 127.0.0.1 --port 8001
```
2. Open `index.html` in a browser (double-click works once the backend is
   running; internet needed once for the Three.js CDN). The badge under the
   Weather Station heading shows whether the backend is online.
3. Optional check: `curl http://127.0.0.1:8001/health` and
   `curl http://127.0.0.1:8001/metrics`.

Verify the model any time with (from `backend`):
```powershell
python verify_model.py
```

Regenerate the chart data (`data.js`) from the notebook pipeline:
```powershell
python build_site_data.py
```
It re-runs the 5 models with the notebook's settings (Random Forest = the
shipped `.pkl`) and refuses to write unless every confusion matrix matches
the notebook output.

## Files
- `index.html`, `styles.css`, `app.js` — page, theme, live predictor
- `charts.js` + `data.js` — notebook figures (target split, rain rate by cloud, all 8 importances, train vs test, ROC curves, correlation heatmap, all 5 confusion matrices, feature summary)
- `sky3d.js` — 3D background and rule cube
- `notebook/Rain_Prediction.ipynb` — the notebook, including the “rain rate by cloud” and “verify the rule” cells the site cites
- `backend/` — FastAPI app, model, scaler artifacts, dataset CSV, verification scripts

## Input validation
`POST /predict` rejects any reading outside the dataset range (HTTP 422),
e.g. Humidity must be 20–100, Rainfall 0–50. The browser clamps typed values
to the same ranges.

## Verified numbers (from the executed notebook)
- 55,996 rows × 9 cols · target RainTomorrow, 18.6% rain (10,417 / 55,996)
- Split 80/20 stratified: 44,796 train / 11,200 test, random_state = 42
- Decision Tree 100.00% / AUC 1.0000 · Random Forest 100.00% / AUC 1.0000
- SVC 97.66% / 0.9965 (trained on 10k subsample) · KNN 95.87% / 0.9879 · Logistic Regression 93.11% / 0.9766
- Confusion (RF): [[9116, 0], [0, 2084]] · Importances: Humidity 0.4862, Cloud 0.4742, Rainfall 0.0273, rest ≈ 0.0025
- Data quirks kept & reported: MinTemp > MaxTemp in 12.4% of rows; WindGustSpeed < WindSpeed in 37.7%
