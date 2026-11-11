from pathlib import Path
import os

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "query (1) (1).csv"
FEATURES = [
    "month",
    "latitude",
    "longitude",
    "depth",
    "magType",
    "nst",
    "gap",
    "dmin",
    "rms",
    "net",
    "place",
]
CATEGORICAL_FEATURES = ("magType", "net", "place")
NUMERIC_FEATURES = ("month", "latitude", "longitude", "depth", "nst", "gap", "dmin", "rms")

app = Flask(__name__)
_magnitude_model = None
_encoders = None
_scaler = None


def _load_training_data():
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"Training dataset not found: {DATASET_PATH.name}. "
            "Place the CSV next to app.py."
        )

    data = pd.read_csv(DATASET_PATH, encoding="latin-1")
    required = {"time", "mag", "magType", "latitude", "longitude", "depth",
                "nst", "gap", "dmin", "rms", "net", "place"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

    data["time"] = pd.to_datetime(data["time"], errors="coerce", utc=True)
    data["month"] = data["time"].dt.month
    data["mag"] = pd.to_numeric(data["mag"], errors="coerce")
    data["place"] = data["place"].fillna("Unknown").astype(str).apply(
        lambda value: value.split(", ")[1] if ", " in value else value
    )

    for column in NUMERIC_FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        if data[column].isna().all():
            raise ValueError(f"Dataset column '{column}' has no usable numeric values.")
        data[column] = data[column].fillna(data[column].mean())

    data = data.dropna(subset=["month", "mag"])
    for column in CATEGORICAL_FEATURES:
        data[column] = data[column].fillna("Unknown").astype(str)
    return data


def _encode_features(data, encoders):
    features = data[FEATURES].copy()
    for column in CATEGORICAL_FEATURES:
        encoder = encoders[column]
        known_values = set(encoder.classes_)
        features[column] = features[column].map(
            lambda value: int(encoder.transform([value])[0]) if value in known_values else -1
        )
    return features.to_numpy(dtype=float)


def _ensure_model():
    global _magnitude_model, _encoders, _scaler
    if _magnitude_model is not None:
        return

    data = _load_training_data()
    _encoders = {}
    for column in CATEGORICAL_FEATURES:
        _encoders[column] = LabelEncoder().fit(data[column])

    X = _encode_features(data, _encoders)
    y = data["mag"].to_numpy(dtype=float)
    _scaler = StandardScaler().fit(X)
    _magnitude_model = LinearRegression().fit(_scaler.transform(X), y)


@app.get("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/api/health")
def health():
    return jsonify(status="ok")


@app.get("/api/options")
def prediction_options():
    try:
        data = _load_training_data()
    except (OSError, ValueError, pd.errors.ParserError) as error:
        return jsonify(error=str(error)), 500

    return jsonify({
        "magType": sorted(data["magType"].dropna().astype(str).unique().tolist()),
        "net": sorted(data["net"].dropna().astype(str).unique().tolist()),
        "place": sorted(data["place"].dropna().astype(str).unique().tolist()),
    })


@app.post("/api/predict")
def predict():
    try:
        payload = request.get_json()
        if not isinstance(payload, dict):
            return jsonify(error="Submit prediction inputs as JSON."), 400

        values = {}
        for field in FEATURES:
            if field not in payload or str(payload[field]).strip() == "":
                return jsonify(error=f"Please provide {field}."), 400
            if field in CATEGORICAL_FEATURES:
                values[field] = str(payload[field]).strip()
            else:
                try:
                    values[field] = float(payload[field])
                except (TypeError, ValueError):
                    return jsonify(error=f"{field} must be a number."), 400
                if not np.isfinite(values[field]):
                    return jsonify(error=f"{field} must be a finite number."), 400

        if not values["month"].is_integer() or not 1 <= values["month"] <= 12:
            return jsonify(error="Month must be a whole number from 1 to 12."), 400
        if not -90 <= values["latitude"] <= 90:
            return jsonify(error="Latitude must be between -90 and 90."), 400
        if not -180 <= values["longitude"] <= 180:
            return jsonify(error="Longitude must be between -180 and 180."), 400
        for field in ("depth", "nst", "gap", "dmin", "rms"):
            if values[field] < 0:
                return jsonify(error=f"{field} cannot be negative."), 400

        _ensure_model()
        features = pd.DataFrame([values], columns=FEATURES)
        encoded = _encode_features(features, _encoders)
        scaled = _scaler.transform(encoded)
        magnitude = float(_magnitude_model.predict(scaled)[0])

        return jsonify(magnitude=round(magnitude, 1))
    except (OSError, ValueError, pd.errors.ParserError) as error:
        return jsonify(error=str(error)), 500


@app.get("/<path:filename>")
def project_file(filename):
    return send_from_directory(BASE_DIR, filename)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5001")),
        debug=False,
    )
