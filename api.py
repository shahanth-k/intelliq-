"""
IntelliQ — FastAPI Backend
===========================
Endpoints:
  GET  /              → serves index.html
  GET  /api/health    → server health + model status
  POST /api/predict   → predict with RF, GB, or both
  GET  /api/metrics   → model comparison metrics from CSV
  GET  /api/summary   → training summary JSON
  GET  /api/dataset   → paginated dataset preview
  GET  /api/dataset/download  → full CSV download
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent

# ── Load models at startup ────────────────────────────────────────────────────
def _load(name: str):
    p = ROOT / "models" / name
    if not p.exists():
        raise RuntimeError(f"Model file not found: {p}")
    return joblib.load(p)

try:
    RF_MODEL = _load("random_forest.joblib")
    GB_MODEL = _load("gradient_boosting.joblib")
    XGB_MODEL = _load("xgboost_model.joblib")
    MODELS_LOADED = True
    LOAD_ERROR = None
except Exception as e:
    RF_MODEL = GB_MODEL = XGB_MODEL = None
    MODELS_LOADED = False
    LOAD_ERROR = str(e)

# ── Load static data ──────────────────────────────────────────────────────────
def _read_metrics():
    p = ROOT / "model_comparison.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p)
    return df.to_dict(orient="records")

def _read_summary():
    p = ROOT / "training_summary.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="IntelliQ API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files from the `static/` folder (CSS, JS, images)
static_dir = ROOT / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    arrival_hour:   int   = Field(..., ge=0, le=23,  description="Hour 0-23")
    arrival_minute: int   = Field(..., ge=0, le=59,  description="Minute 0-59")
    queue_length:   float = Field(..., ge=0, le=200, description="Queue length ≥ 0")
    model:          str   = Field("all", description="'random_forest' | 'gradient_boosting' | 'xgboost' | 'all'")

class PredictResponse(BaseModel):
    random_forest:      float | None = None
    gradient_boosting:  float | None = None
    xgboost:            float | None = None
    model_used:         str
    inputs: dict

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_frontend():
    html_path = ROOT / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>index.html not found. Run build step.</h1>", status_code=404)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "models_loaded": MODELS_LOADED,
        "load_error": LOAD_ERROR,
        "models": ["random_forest", "gradient_boosting", "xgboost"] if MODELS_LOADED else [],
    }

@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not MODELS_LOADED:
        raise HTTPException(503, detail=f"Models not loaded: {LOAD_ERROR}")

    X = pd.DataFrame([{
        "arrival_hour":   req.arrival_hour,
        "arrival_minute": req.arrival_minute,
        "queue_length":   req.queue_length,
    }])

    model_key = req.model.lower().replace(" ", "_")
    rf_pred = None
    gb_pred = None
    xgb_pred = None

    if model_key in ("random_forest", "both", "all", "compare"):
        rf_pred = round(float(max(0.0, RF_MODEL.predict(X)[0])), 2)
    if model_key in ("gradient_boosting", "both", "all", "compare"):
        gb_pred = round(float(max(0.0, GB_MODEL.predict(X)[0])), 2)
    if model_key in ("xgboost", "all", "compare"):
        xgb_pred = round(float(max(0.0, XGB_MODEL.predict(X)[0])), 2)

    return PredictResponse(
        random_forest=rf_pred,
        gradient_boosting=gb_pred,
        xgboost=xgb_pred,
        model_used=req.model,
        inputs={
            "arrival_hour": req.arrival_hour,
            "arrival_minute": req.arrival_minute,
            "queue_length": req.queue_length,
        },
    )

@app.post("/predict")
async def simple_predict(req: dict):
    if not MODELS_LOADED:
        raise HTTPException(503, detail=f"Models not loaded: {LOAD_ERROR}")
    
    arrival_time_str = req.get("arrival_time")
    queue_length_val = req.get("queue_length")
    model_choice = req.get("model", "Random Forest")

    if not arrival_time_str or queue_length_val is None:
        raise HTTPException(400, "Please provide arrival_time and queue_length.")

    try:
        parts = str(arrival_time_str).strip().split(":")
        arrival_hour = int(parts[0])
        arrival_minute = int(parts[1]) if len(parts) > 1 else 0
        queue_length = float(queue_length_val)
    except Exception as e:
        raise HTTPException(400, f"Invalid inputs: {e}")

    X = pd.DataFrame([{
        "arrival_hour": arrival_hour,
        "arrival_minute": arrival_minute,
        "queue_length": queue_length,
    }])

    choice_lower = model_choice.lower()
    if "xgb" in choice_lower:
        pred = float(XGB_MODEL.predict(X)[0])
        return {"model": "XGBoost", "prediction": round(pred, 2)}
    elif "gradient" in choice_lower:
        pred = float(GB_MODEL.predict(X)[0])
        return {"model": "Gradient Boosting", "prediction": round(pred, 2)}
    elif any(term in choice_lower for term in ["compare", "all", "both"]):
        rf_pred = float(RF_MODEL.predict(X)[0])
        gb_pred = float(GB_MODEL.predict(X)[0])
        xgb_pred = float(XGB_MODEL.predict(X)[0])
        return {
            "model": "Compare All Three Models",
            "random_forest": round(rf_pred, 2),
            "gradient_boosting": round(gb_pred, 2),
            "xgboost": round(xgb_pred, 2),
        }
    else:
        pred = float(RF_MODEL.predict(X)[0])
        return {"model": "Random Forest", "prediction": round(pred, 2)}


@app.get("/api/metrics")
def metrics():
    data = _read_metrics()
    if not data:
        raise HTTPException(404, "model_comparison.csv not found. Run train_models.py first.")
    return {"metrics": data}

@app.get("/api/summary")
def summary():
    data = _read_summary()
    if not data:
        raise HTTPException(404, "training_summary.json not found.")
    return data

@app.get("/api/dataset")
def dataset(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    search: str = Query("", description="Filter rows where any column contains this string"),
):
    csv_path = ROOT / "queue_data.csv"
    if not csv_path.exists():
        raise HTTPException(404, "queue_data.csv not found.")
    df = pd.read_csv(csv_path)
    # Keep only the five canonical columns
    cols = [c for c in ["arrival_time", "start_time", "finish_time", "wait_time", "queue_length"] if c in df.columns]
    df = df[cols]
    if search.strip():
        mask = df.apply(lambda col: col.astype(str).str.contains(search, case=False, na=False)).any(axis=1)
        df = df[mask]
    total = len(df)
    start = (page - 1) * page_size
    page_df = df.iloc[start: start + page_size]
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": max(1, -(-total // page_size)),
        "columns": list(page_df.columns),
        "rows": page_df.fillna("").to_dict(orient="records"),
    }

@app.get("/api/dataset/download")
def dataset_download():
    csv_path = ROOT / "queue_data.csv"
    if not csv_path.exists():
        raise HTTPException(404, "queue_data.csv not found.")
    return FileResponse(
        str(csv_path),
        media_type="text/csv",
        filename="intelliq_queue_data.csv",
    )

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
