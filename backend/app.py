"""
DS422 — Weather-Based Rain Prediction: FastAPI backend for ATMOS.

Loads random_forest_rain_model.pkl (joblib) ONCE at startup, server-side
only, and serves live predictions that replicate the Colab notebook's
preprocessing exactly:

  Features (notebook order, X = df.drop(columns=["RainTomorrow"])):
    MinTemp, MaxTemp, Rainfall, Humidity, Pressure, WindSpeed,
    WindGustSpeed, Cloud
  StandardScaler on ALL 8 features, fitted on the train split ONLY
  (train_test_split test_size=0.2, random_state=42, stratify=y;
   44,796 train / 11,200 test).

IMPORTANT: the .pkl contains ONLY the trained Random Forest model — not
the scaler. The scaler is loaded from the verified
preprocessing_artifacts.json in this folder (produced by
verify_model.py, which confirms the loaded model + rebuilt scaler
reproduce 100.00% accuracy, ROC-AUC 1.0 and confusion
[[9116, 0], [0, 2084]] on the full test set). If that file is missing,
the scaler is rebuilt from the dataset CSV at startup using the exact
split above.

Configuration (environment variables):
  MODEL_PATH    Path to random_forest_rain_model.pkl
                (default: ./random_forest_rain_model.pkl, then
                 ~/workspace/ds422_lab/rain/random_forest_rain_model.pkl)
  DATA_CSV_PATH Path to weather_based_rain_prediction_dataset.csv
                (only needed if preprocessing_artifacts.json is absent)

Run (port 8001 — churn backend uses 8000), from this backend folder:
  python -m uvicorn app:app --host 127.0.0.1 --port 8001
"""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = ["MinTemp", "MaxTemp", "Rainfall", "Humidity", "Pressure",
            "WindSpeed", "WindGustSpeed", "Cloud"]

# Exact min / max of every feature in the dataset (notebook cell 7, df.describe()).
FEATURE_RANGES = {
    "MinTemp": [0.0, 30.0],
    "MaxTemp": [15.0, 45.0],
    "Rainfall": [0.0, 50.0],
    "Humidity": [20.0, 100.0],
    "Pressure": [980.0, 1035.0],
    "WindSpeed": [0.0, 60.0],
    "WindGustSpeed": [0.0, 80.0],
    "Cloud": [0.0, 8.0],
}

# Thresholds learned by the unscaled display tree (notebook cell 24).
TREE_THRESHOLDS = {"Humidity": 65.01, "Cloud": 4.5, "Rainfall": 2.01}

METRICS = {
    "best_model": "Random Forest",
    "models": [
        {"model": "Decision Tree", "train_accuracy": 1.0, "test_accuracy": 1.0, "roc_auc": 1.0,
         "confusion_matrix": [[9116, 0], [0, 2084]]},
        {"model": "Random Forest", "train_accuracy": 1.0, "test_accuracy": 1.0, "roc_auc": 1.0,
         "confusion_matrix": [[9116, 0], [0, 2084]]},
        {"model": "SVC", "train_accuracy": 0.9809, "test_accuracy": 0.9766, "roc_auc": 0.9965,
         "confusion_matrix": [[9007, 109], [153, 1931]]},
        {"model": "KNN", "train_accuracy": 0.9765, "test_accuracy": 0.9587, "roc_auc": 0.9879,
         "confusion_matrix": [[8912, 204], [259, 1825]]},
        {"model": "Logistic Regression", "train_accuracy": 0.934, "test_accuracy": 0.9311, "roc_auc": 0.9766,
         "confusion_matrix": [[8770, 346], [426, 1658]]},
    ],
    "random_forest": {
        "test_accuracy": 1.0,
        "roc_auc": 1.0,
        "rain_precision": 1.0,
        "rain_recall": 1.0,
        "rain_f1": 1.0,
        "confusion_matrix": [[9116, 0], [0, 2084]],
    },
    "tree_thresholds": TREE_THRESHOLDS,
    "feature_ranges": FEATURE_RANGES,
    "feature_importances": [
        {"feature": "Humidity", "importance": 0.4862},
        {"feature": "Cloud", "importance": 0.4742},
        {"feature": "Rainfall", "importance": 0.0273},
        {"feature": "WindSpeed", "importance": 0.0026},
        {"feature": "Pressure", "importance": 0.0025},
        {"feature": "MinTemp", "importance": 0.0025},
        {"feature": "WindGustSpeed", "importance": 0.0024},
        {"feature": "MaxTemp", "importance": 0.0024},
    ],
    "discovered_rule": "Humidity > 65 AND Cloud > 4.5 AND Rainfall > 2",
    "dataset": {
        "rows": 55996, "columns": 9, "features": 8,
        "rain_days": 10417, "dry_days": 45579, "rain_rate": 0.186,
        "train_rows": 44796, "test_rows": 11200,
    },
}

STATE = {"model": None, "scaler": None, "startup_error": None,
         "preprocessing_source": None}


def _first_existing(env_var, candidates):
    env = os.environ.get(env_var)
    if env:
        return Path(env).expanduser()
    for c in candidates:
        p = Path(c).expanduser()
        if p.exists():
            return p
    return Path(candidates[0]).expanduser()


def _scaler_from_artifacts(path: Path) -> StandardScaler:
    data = json.loads(path.read_text())
    if data.get("feature_columns") != FEATURES:
        raise ValueError("Artifacts feature_columns do not match FEATURES")
    scaler = StandardScaler()
    scaler.mean_ = np.array([data["scaler_mean"][f] for f in FEATURES])
    scaler.scale_ = np.array([data["scaler_scale"][f] for f in FEATURES])
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(FEATURES)
    scaler.feature_names_in_ = np.array(FEATURES)
    scaler.n_samples_seen_ = int(data.get("scaler_n_samples_seen", 44796))
    STATE["preprocessing_source"] = f"verified artifacts: {path.name}"
    return scaler


def _build_scaler(csv_path: Path) -> StandardScaler:
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["RainTomorrow"])
    y = df["RainTomorrow"]
    X_train, _X_test, _y_train, _y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    scaler.fit(X_train[FEATURES])
    STATE["preprocessing_source"] = f"rebuilt from CSV: {csv_path}"
    return scaler


def _load_all() -> None:
    try:
        model_path = _first_existing("MODEL_PATH", [
            "./random_forest_rain_model.pkl",
            "~/workspace/ds422_lab/rain/random_forest_rain_model.pkl",
        ])
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        artifacts = Path(__file__).resolve().parent / "preprocessing_artifacts.json"
        if artifacts.exists():
            STATE["scaler"] = _scaler_from_artifacts(artifacts)
        else:
            csv_path = _first_existing("DATA_CSV_PATH", [
                "./weather_based_rain_prediction_dataset.csv",
                "~/workspace/ds422_lab/rain/weather_based_rain_prediction_dataset.csv",
            ])
            if not csv_path.exists():
                raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")
            STATE["scaler"] = _build_scaler(csv_path)
        STATE["model"] = joblib.load(model_path)  # loaded once, server-side only
        STATE["model_path"] = str(model_path)
    except Exception as exc:
        STATE["startup_error"] = f"{type(exc).__name__}: {exc}"


app = FastAPI(title="DS422 Rain Prediction API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    _load_all()


class WeatherInput(BaseModel):
    # Bounds = exact min/max of each column in the 55,996-row dataset.
    # Values outside them were never seen in training, so they are rejected
    # (HTTP 422) instead of producing a confident but meaningless answer.
    MinTemp: float = Field(..., ge=FEATURE_RANGES["MinTemp"][0], le=FEATURE_RANGES["MinTemp"][1])
    MaxTemp: float = Field(..., ge=FEATURE_RANGES["MaxTemp"][0], le=FEATURE_RANGES["MaxTemp"][1])
    Rainfall: float = Field(..., ge=FEATURE_RANGES["Rainfall"][0], le=FEATURE_RANGES["Rainfall"][1])
    Humidity: float = Field(..., ge=FEATURE_RANGES["Humidity"][0], le=FEATURE_RANGES["Humidity"][1])
    Pressure: float = Field(..., ge=FEATURE_RANGES["Pressure"][0], le=FEATURE_RANGES["Pressure"][1])
    WindSpeed: float = Field(..., ge=FEATURE_RANGES["WindSpeed"][0], le=FEATURE_RANGES["WindSpeed"][1])
    WindGustSpeed: float = Field(..., ge=FEATURE_RANGES["WindGustSpeed"][0], le=FEATURE_RANGES["WindGustSpeed"][1])
    Cloud: float = Field(..., ge=FEATURE_RANGES["Cloud"][0], le=FEATURE_RANGES["Cloud"][1])


@app.get("/health")
def health():
    ok = STATE["model"] is not None and STATE["scaler"] is not None
    return {
        "status": "ok" if ok else "degraded",
        "model_loaded": STATE["model"] is not None,
        "model_path": STATE.get("model_path"),
        "model_type": type(STATE["model"]).__name__ if STATE["model"] is not None else None,
        "n_estimators": getattr(STATE["model"], "n_estimators", None),
        "scaler_ready": STATE["scaler"] is not None,
        "preprocessing_source": STATE.get("preprocessing_source"),
        "sklearn_version": sklearn.__version__,
        "feature_ranges": FEATURE_RANGES,
        "startup_error": STATE["startup_error"],
    }


@app.get("/metrics")
def metrics():
    return METRICS


@app.post("/predict")
def predict(weather: WeatherInput):
    if STATE["model"] is None or STATE["scaler"] is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not available. Startup error: {STATE['startup_error']}",
        )
    row = pd.DataFrame(
        [{f: float(getattr(weather, f)) for f in FEATURES}], columns=FEATURES
    )
    X = pd.DataFrame(
        STATE["scaler"].transform(row[FEATURES]), columns=FEATURES
    )
    prediction = int(STATE["model"].predict(X)[0])
    probability = float(STATE["model"].predict_proba(X)[0][1])
    # Distance of each rule variable from the learned tree threshold
    # (positive = gate passed). Explanation only — the verdict is the RF's.
    distances = {f: round(float(getattr(weather, f)) - t, 2) for f, t in TREE_THRESHOLDS.items()}
    return {
        "prediction": prediction,
        "label": "RAIN" if prediction == 1 else "NO RAIN",
        "rain_probability": round(probability, 4),
        "rain_probability_percent": round(probability * 100, 1),
        "trees_voting_rain": int(round(probability * STATE["model"].n_estimators)),
        "n_estimators": int(STATE["model"].n_estimators),
        "threshold_distance": distances,
        "model": "Random Forest (random_forest_rain_model.pkl)",
    }
