"""
Verify random_forest_rain_model.pkl reproduces the notebook exactly, and
write preprocessing_artifacts.json (scaler rebuilt as in the notebook).

Notebook pipeline (DS422_Rain_Prediction.ipynb):
  X = df.drop(columns=["RainTomorrow"])   # CSV order, 8 numeric features
  train_test_split(test_size=0.2, random_state=42, stratify=y)
  StandardScaler fitted on X_train ONLY, applied to all 8 features
  RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

The .pkl contains ONLY the trained Random Forest — no scaler — so the
scaler must be rebuilt from the dataset CSV with the exact split above.

Run (from this folder, using the ds422_lab venv):
  ~/workspace/ds422_lab/.venv/bin/python verify_model.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    HERE / "random_forest_rain_model.pkl",
    Path.home() / "workspace/ds422_lab/rain/random_forest_rain_model.pkl",
]
CSV_CANDIDATES = [
    HERE / "weather_based_rain_prediction_dataset.csv",
    Path.home() / "workspace/ds422_lab/rain/weather_based_rain_prediction_dataset.csv",
    Path.home() / "workspace/user/files/weather_based_rain_prediction_dataset.csv",
]

FEATURES = ["MinTemp", "MaxTemp", "Rainfall", "Humidity", "Pressure",
            "WindSpeed", "WindGustSpeed", "Cloud"]


def _first(paths):
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(f"None of these exist: {paths}")


def main():
    model_path = _first(MODEL_CANDIDATES)
    csv_path = _first(CSV_CANDIDATES)
    print(f"Model: {model_path}")
    print(f"CSV:   {csv_path}")

    model = joblib.load(model_path)
    print(f"Type: {type(model).__name__}")
    print(f"n_features_in_: {getattr(model, 'n_features_in_', None)}")
    names = getattr(model, "feature_names_in_", None)
    print(f"feature_names_in_: {list(names) if names is not None else None}")
    print(f"classes_: {list(model.classes_)}")
    print(f"feature_importances_: {[round(float(x), 6) for x in model.feature_importances_]}")
    assert list(names) == FEATURES, "Model feature order does not match notebook order"

    df = pd.read_csv(csv_path)
    X = df.drop(columns=["RainTomorrow"])
    assert X.columns.tolist() == FEATURES
    y = df["RainTomorrow"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    scaler.fit(X_train[FEATURES])

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test[FEATURES]), columns=FEATURES, index=X_test.index
    )
    pred = model.predict(X_test_scaled)
    proba = model.predict_proba(X_test_scaled)[:, 1]
    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)
    cm = confusion_matrix(y_test, pred).tolist()
    first_pred = int(pred[0])
    first_actual = int(y_test.iloc[0])

    print(f"Test rows: {len(y_test)}  Train rows: {len(y_train)}")
    print(f"Accuracy: {acc:.4f}  ROC-AUC: {auc:.4f}")
    print(f"Confusion matrix: {cm}")
    print(f"First test row: prediction={first_pred} actual={first_actual} proba_rain={float(proba[0]):.4f}")

    assert acc == 1.0 and auc == 1.0 and cm == [[9116, 0], [0, 2084]], \
        "Verification FAILED — does not reproduce the notebook"
    assert first_pred == first_actual == 0
    print("VERIFIED: reproduces notebook exactly (100.00%, AUC 1.0, [[9116,0],[0,2084]])")

    artifacts = {
        "feature_columns": FEATURES,
        "scaler_mean": {f: float(scaler.mean_[i]) for i, f in enumerate(FEATURES)},
        "scaler_scale": {f: float(scaler.scale_[i]) for i, f in enumerate(FEATURES)},
        "scaler_n_samples_seen": int(scaler.n_samples_seen_),
        "split": {"test_size": 0.2, "random_state": 42, "stratify": True,
                  "train_rows": int(len(y_train)), "test_rows": int(len(y_test))},
        "verified": {"test_accuracy": float(acc), "roc_auc": float(auc),
                     "confusion_matrix": cm,
                     "first_test_prediction": first_pred,
                     "first_test_actual": first_actual},
        "model_file": model_path.name,
    }
    out = HERE / "preprocessing_artifacts.json"
    out.write_text(json.dumps(artifacts, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
