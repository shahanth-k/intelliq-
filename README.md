# IntelliQ — Adaptive AI-Based Queue Prediction System

**Predicts customer waiting times for retail and supermarket checkout queues using Random Forest, Gradient Boosting, and XGBoost Regressors.**

---

## Quick Start (Windows / VS Code)

### 1. Install dependencies
Open PowerShell in the project directory (`d:\last commit\IntelliQ_Project`):
```powershell
pip install -r requirements.txt
```

### 2. Train and evaluate all three models
Trains Random Forest, Gradient Boosting, and XGBoost on the same validated dataset split, evaluates on held-out test records, and saves the joblib models:
```powershell
python train_models.py
```

### 3. Start the backend server
Runs the Flask application serving both the frontend webpage and the prediction API:
```powershell
python server.py
```

### 4. Open the web app
Open your browser and navigate to:
**http://127.0.0.1:5000**

---

## What the App Predicts

You enter:
- **Arrival Time** — when you arrive at the store (e.g., `12:00` or `16:47`)
- **Queue Length** — total number of people currently waiting in queue (e.g., `5`, `10`, `20`, `30`)
- **Model Selection** — choose between:
  - `Random Forest`
  - `Gradient Boosting`
  - `XGBoost`
  - `Compare All Three Models`

Click **Predict Waiting Time**:
- If a single model is selected, its estimated waiting time (in minutes), confidence score, people ahead, and expected service start time are displayed.
- If **Compare All Three Models** is selected, predictions from all three algorithms are displayed side by side.

---

## Project Structure

```
IntelliQ_Project/
├── queue_data.csv                    ← Cleaned dataset (1,567 rows, 5 columns)
├── queue_data_backup_before_xgb.csv  ← Dataset backup before XGBoost integration
├── queue_data_backup.csv             ← Additional backup copy
├── queue_data_original.csv           ← Original backup copy
├── train_models.py                   ← Model training and evaluation pipeline
├── server.py                         ← Flask backend (/predict API + frontend)
├── index.html                        ← Single-page frontend
├── models/
│   ├── random_forest.joblib          ← Trained Random Forest Regressor
│   ├── gradient_boosting.joblib      ← Trained Gradient Boosting Regressor
│   └── xgboost_model.joblib          ← Trained XGBoost Regressor
├── model_comparison.csv              ← Evaluation metrics (MAE, RMSE, R²) for all 3 models
├── test_predictions.csv              ← Held-out test set predictions
├── training_summary.json             ← JSON summary of pipeline and metrics
├── training_output.txt               ← Plaintext summary of evaluation
├── requirements.txt                  ← Dependencies (flask, pandas, numpy, scikit-learn, joblib, xgboost)
└── README.md                         ← Run instructions and documentation
```

---

## Model Evaluation Results

All three models were evaluated on the exact same 20% held-out test split (314 rows out of 1,567 total records):

| Model | MAE (minutes) | RMSE (minutes) | R² Score | Best Performing |
|---|---|---|---|---|
| **Random Forest** | 0.190 | 0.291 | 1.000 | |
| **Gradient Boosting** | 0.143 | 0.190 | 1.000 | |
| **XGBoost** | **0.134** | **0.179** | **1.000** | **✓ Lowest MAE & RMSE** |

### Did XGBoost improve performance?
**Yes.** On the held-out test records:
- XGBoost reduced MAE to **0.134 minutes** (compared with 0.143 min for Gradient Boosting and 0.190 min for Random Forest).
- XGBoost reduced RMSE to **0.179 minutes** (compared with 0.190 min for Gradient Boosting and 0.291 min for Random Forest).
- XGBoost achieved the best overall predictive accuracy among all three algorithms.

---

## Queue Length Sensitivity Test (Arrival Time: 12:00)

Testing predictions across queue lengths of 5, 10, 20, and 30:

| Queue Length | Random Forest (min) | Gradient Boosting (min) | XGBoost (min) | Behavior / Note |
|---|---|---|---|---|
| **5** | 6.8 min | 6.9 min | 6.9 min | Small queue (~4 people ahead) |
| **10** | 14.3 min | 14.3 min | 14.3 min | Moderate queue (~9 people ahead) |
| **20** | 28.9 min | 28.9 min | 28.9 min | Busy checkout (~19 people ahead) |
| **30** | 43.6 min | 43.7 min | 43.7 min | Peak rush queue (~29 people ahead) |

**Analysis:**
All three algorithms scale consistently and smoothly with queue length at approximately **1.45–1.50 minutes per person ahead in line**, with slight adjustments for store rush-hour patterns. Predictions are consistent across all three models.

---

## API Reference

### `POST /predict` (also available as `POST /api/predict`)

**Request body (JSON):**
```json
{
  "arrival_time": "12:00",
  "queue_length": 5,
  "model": "XGBoost"
}
```

Available `model` values:
- `"Random Forest"`
- `"Gradient Boosting"`
- `"XGBoost"`
- `"Compare All Three Models"`

**Single Model Response (e.g. XGBoost):**
```json
{
  "status": "success",
  "model": "XGBoost",
  "is_comparison": false,
  "prediction": 6.9,
  "people_ahead": 4,
  "range_min": 5,
  "range_max": 9,
  "expected_service_time": "12:07 PM (estimated)",
  "service_time_short": "12:07 PM",
  "confidence": 92,
  "queue_length": 5
}
```

**Compare All Three Models Response:**
```json
{
  "status": "success",
  "model": "Compare All Three Models",
  "is_comparison": true,
  "prediction": 6.9,
  "predictions": {
    "random_forest": 6.8,
    "gradient_boosting": 6.9,
    "xgboost": 6.9
  },
  "people_ahead": 4,
  "range_min": 6,
  "range_max": 8,
  "expected_service_time": "12:07 PM (estimated)",
  "service_time_short": "12:07 PM",
  "confidence": 92,
  "queue_length": 5
}
```

---

## Troubleshooting

| Issue | Resolution |
|---|---|
| `ModuleNotFoundError: No module named 'xgboost'` | Run `pip install xgboost` or `pip install -r requirements.txt` |
| `models/xgboost_model.joblib not found` | Run `python train_models.py` |
| Browser shows connection refused | Verify `server.py` is running in your terminal |
| Port 5000 in use | In `server.py`, change `port=5000` to `port=5001` |
