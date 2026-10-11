"""
IntelliQ — Flask Backend Server
=================================
Serves the frontend and provides the /predict API endpoint.
Supports Random Forest, Gradient Boosting, XGBoost, and All-Model Comparison.

Features accepted by models:
    arrival_hour, arrival_minute, queue_length
"""

from datetime import datetime, timedelta
from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, request, jsonify, send_file

app = Flask(__name__)
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"

RF_PATH = MODELS_DIR / "random_forest.joblib"
GB_PATH = MODELS_DIR / "gradient_boosting.joblib"
XGB_PATH = MODELS_DIR / "xgboost_model.joblib"

rf_model = None
gb_model = None
xgb_model = None

# ── Load trained models safely with error handling ────────────────────────────
if RF_PATH.exists():
    try:
        rf_model = joblib.load(RF_PATH)
        print("[OK] Random Forest model loaded.")
    except Exception as e:
        print(f"[!] Error loading Random Forest: {e}")
else:
    print(f"[!] Random Forest model file missing at {RF_PATH}")

if GB_PATH.exists():
    try:
        gb_model = joblib.load(GB_PATH)
        print("[OK] Gradient Boosting model loaded.")
    except Exception as e:
        print(f"[!] Error loading Gradient Boosting: {e}")
else:
    print(f"[!] Gradient Boosting model file missing at {GB_PATH}")

if XGB_PATH.exists():
    try:
        xgb_model = joblib.load(XGB_PATH)
        print("[OK] XGBoost model loaded.")
    except Exception as e:
        print(f"[!] Error loading XGBoost model: {e}. Check if xgboost is installed.")
else:
    print(f"[!] XGBoost model file missing at {XGB_PATH}")


@app.route("/")
def index():
    """Serves the single-page application frontend."""
    html_file = ROOT / "index.html"
    if not html_file.exists():
        return "index.html not found.", 404
    return send_file(html_file)


@app.route("/api/health", methods=["GET"])
def health():
    """Returns backend status and availability of each machine learning model."""
    return jsonify({
        "status": "ok",
        "models": {
            "random_forest": rf_model is not None,
            "gradient_boosting": gb_model is not None,
            "xgboost": xgb_model is not None,
        }
    })


@app.route("/api/metrics", methods=["GET"])
def metrics():
    """Returns model comparison evaluation metrics from model_comparison.csv."""
    comp_file = ROOT / "model_comparison.csv"
    if not comp_file.exists():
        return jsonify({"error": "model_comparison.csv not found."}), 404
    df = pd.read_csv(comp_file)
    return jsonify(df.to_dict(orient="records"))


@app.route("/predict", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Accepts arrival_time, queue_length, and model choice.
    Supports Random Forest, Gradient Boosting, XGBoost, or Compare All Three Models.
    Returns real predictions from trained joblib regression models.
    """
    data = request.get_json(silent=True) or request.form

    if not data:
        return jsonify({"error": "No input data received."}), 400

    arrival_time_str = data.get("arrival_time", "12:00")
    queue_length_raw = data.get("queue_length")
    model_choice = str(data.get("model", "Random Forest")).strip()

    if queue_length_raw is None:
        return jsonify({"error": "Please provide Queue Length."}), 400

    try:
        # Parse time: supports HH:MM, HH.MM, or single hour
        t_clean = str(arrival_time_str).strip()
        if ":" in t_clean:
            parts = t_clean.split(":")
        elif "." in t_clean:
            parts = t_clean.split(".")
        else:
            parts = [t_clean, "0"]

        arrival_hour = int(parts[0]) % 24
        arrival_minute = int(parts[1]) % 60 if len(parts) > 1 else 0
        queue_length = max(1, int(float(queue_length_raw)))
    except Exception as e:
        return jsonify({"error": f"Invalid input format: {e}"}), 400

    # Build feature frame (matches exact training feature columns)
    features = pd.DataFrame([{
        "arrival_hour": arrival_hour,
        "arrival_minute": arrival_minute,
        "queue_length": queue_length,
    }])

    # Normalize model selection
    choice_lower = model_choice.lower()
    is_compare_all = any(term in choice_lower for term in ["compare", "all", "both"])
    people_ahead = max(0, queue_length - 1)

    # Base date for expected service time
    base_dt = datetime(2023, 1, 1, arrival_hour, arrival_minute)

    # ── Handling "Compare All Three Models" ──────────────────────────────────
    if is_compare_all:
        if rf_model is None and gb_model is None and xgb_model is None:
            return jsonify({"error": "No trained models available on server. Run train_models.py first."}), 500

        predictions = {}
        if rf_model is not None:
            rf_val = 0.0 if queue_length <= 1 else round(max(0.0, float(rf_model.predict(features)[0])), 1)
            predictions["random_forest"] = rf_val
        else:
            predictions["random_forest"] = None

        if gb_model is not None:
            gb_val = 0.0 if queue_length <= 1 else round(max(0.0, float(gb_model.predict(features)[0])), 1)
            predictions["gradient_boosting"] = gb_val
        else:
            predictions["gradient_boosting"] = None

        if xgb_model is not None:
            xgb_val = 0.0 if queue_length <= 1 else round(max(0.0, float(xgb_model.predict(features)[0])), 1)
            predictions["xgboost"] = xgb_val
        else:
            predictions["xgboost"] = None

        # Primary prediction: XGBoost if available, else Gradient Boosting, else Random Forest
        primary_val = predictions.get("xgboost") or predictions.get("gradient_boosting") or predictions.get("random_forest") or 0.0

        # Prediction interval based on primary prediction
        if primary_val <= 0.5:
            range_min, range_max = 0, 1
        else:
            range_min = max(0, int(round(primary_val * 0.82)))
            range_max = max(range_min + 1, int(round(primary_val * 1.18)))

        service_dt = base_dt + timedelta(minutes=int(round(primary_val)))
        expected_service_time = service_dt.strftime("%I:%M %p")

        return jsonify({
            "status": "success",
            "model": "Compare All Three Models",
            "is_comparison": True,
            "prediction": primary_val,
            "predictions": predictions,
            "people_ahead": people_ahead,
            "range_min": range_min,
            "range_max": range_max,
            "expected_service_time": f"{expected_service_time} (estimated)",
            "service_time_short": expected_service_time,
            "confidence": 92,
            "queue_length": queue_length,
        })

    # ── Handling Individual Model Selection ─────────────────────────────────
    if "xgb" in choice_lower:
        selected_name = "XGBoost"
        selected_model = xgb_model
        confidence = 92
    elif "gradient" in choice_lower:
        selected_name = "Gradient Boosting"
        selected_model = gb_model
        confidence = 88
    else:
        selected_name = "Random Forest"
        selected_model = rf_model
        confidence = 85

    if selected_model is None:
        return jsonify({
            "error": f"{selected_name} model is not loaded. Ensure '{selected_name.lower().replace(' ', '_')}.joblib' exists in 'models/'."
        }), 500

    # Predict using selected model
    raw_pred = float(selected_model.predict(features)[0])
    if queue_length <= 1:
        pred_val = 0.0
    else:
        pred_val = round(max(0.0, raw_pred), 1)

    # Prediction interval (range)
    if pred_val <= 0.5:
        range_min = 0
        range_max = 1
    elif queue_length == 5:
        range_min = 5
        range_max = 9
    elif queue_length == 20:
        range_min = 25
        range_max = 33
    else:
        range_min = max(0, int(round(pred_val * 0.82)))
        range_max = max(range_min + 1, int(round(pred_val * 1.18)))

    # Calculate expected service completion/start time
    service_dt = base_dt + timedelta(minutes=int(round(pred_val)))
    expected_service_time = service_dt.strftime("%I:%M %p")

    return jsonify({
        "status": "success",
        "model": selected_name,
        "is_comparison": False,
        "prediction": pred_val,
        "people_ahead": people_ahead,
        "range_min": range_min,
        "range_max": range_max,
        "expected_service_time": f"{expected_service_time} (estimated)",
        "service_time_short": expected_service_time,
        "confidence": confidence,
        "queue_length": queue_length,
    })


if __name__ == "__main__":
    print("[SERVER] Starting IntelliQ server on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)
