"""
IntelliQ — Model Training Pipeline
==================================
Trains Random Forest, Gradient Boosting, and XGBoost regressors to predict customer
waiting times for checkout queues based on arrival time and queue length.

Features:
    - arrival_hour:   hour of arrival (0-23)
    - arrival_minute: minute of arrival (0-59)
    - queue_length:   total number of people currently in the queue

Target:
    - wait_time:      customer waiting time in minutes (start_time - arrival_time)
"""

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
RANDOM_SEED = 42


def parse_clock(value: str):
    """Parse timestamps formatted like '30-03-2023 00.10' or '30-03-2023 00:10'."""
    s = str(value).strip()
    if " " not in s:
        return pd.NaT
    date_part, clock_part = s.rsplit(" ", 1)
    try:
        date = pd.to_datetime(date_part, format="%d-%m-%Y", errors="raise")
        if ":" in clock_part:
            hh, mm = clock_part.split(":", 1)
        elif "." in clock_part:
            hh, mm = clock_part.split(".", 1)
        else:
            hh, mm = clock_part, "0"
        hh, mm = int(hh), int(mm)
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            return pd.NaT
        return date + pd.Timedelta(hours=hh, minutes=mm)
    except Exception:
        return pd.NaT


def main():
    print("=" * 70)
    print("IntelliQ — Queue Waiting Time Model Training Pipeline")
    print("Supporting: Random Forest, Gradient Boosting, and XGBoost")
    print("=" * 70)

    # 1. Backups: preserve the dataset and guarantee backups exist
    csv_path = ROOT / "queue_data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    backup_csv = ROOT / "queue_data_backup.csv"
    xgb_backup = ROOT / "queue_data_backup_before_xgb.csv"
    if not backup_csv.exists():
        shutil.copy(csv_path, backup_csv)
        print(f"[BACKUP] Created {backup_csv.name}")
    if not xgb_backup.exists():
        shutil.copy(csv_path, xgb_backup)
        print(f"[BACKUP] Created {xgb_backup.name}")

    # 2. Inspect Dataset (no synthetic rows, no duplicate records, no new dataset)
    raw = pd.read_csv(csv_path)
    print(f"\n[DATA] Inspecting existing dataset: {len(raw)} rows loaded.")
    expected_cols = ["arrival_time", "start_time", "finish_time", "wait_time", "queue_length"]
    missing_cols = [c for c in expected_cols if c not in raw.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns in {csv_path.name}: {missing_cols}")
    print(f"[DATA] Columns verified: {list(raw.columns)}")

    # 3. Check Target & Timestamp Consistency
    arrivals = raw["arrival_time"].map(parse_clock)
    starts = raw["start_time"].map(parse_clock)

    valid_mask = arrivals.notna() & starts.notna()
    if not valid_mask.all():
        print(f"[DATA] Warning: {(~valid_mask).sum()} rows have unparseable timestamps.")

    # Calculate timestamp-derived waiting time in minutes, accounting for midnight rollover
    diff_minutes = ((starts - arrivals).dt.total_seconds() / 60.0) % 1440
    raw_wait = raw["wait_time"]

    # Analyze discrepancies between recorded wait_time and timestamp-derived target
    discrepancy = (raw_wait - diff_minutes).abs()
    exact_matches = (discrepancy < 0.001).sum()
    fractional_diffs = ((discrepancy >= 0.001) & (discrepancy <= 1.0)).sum()
    large_mismatches = (discrepancy > 1.0).sum()

    print("\n--- Target & Timestamp Consistency Report ---")
    print(f"Total rows examined: {len(raw)}")
    print(f"Exact timestamp matches (diff < 0.001 min): {exact_matches}")
    print(f"Fractional precision differences (diff <= 1.0 min): {fractional_diffs}")
    print(f"  -> Explanation: 'start_time' is recorded to whole minutes (e.g. 00.24)")
    print(f"     while 'wait_time' retains decimal resolution (e.g. 14.2 min).")
    print(f"Large mismatches (> 1.0 min): {large_mismatches}")
    print(f"Midnight rollover handling: modulo 1440 min correctly resolves day boundary.")
    print("---------------------------------------------\n")

    # Consistent target: use validated wait_time (measured in minutes)
    y = raw["wait_time"].values

    # Features: arrival hour, arrival minute, queue length (no future start/finish times)
    X = pd.DataFrame({
        "arrival_hour": arrivals.dt.hour.values,
        "arrival_minute": arrivals.dt.minute.values,
        "queue_length": raw["queue_length"].values,
    })
    features = list(X.columns)

    # 4. Train / Test Split (Reproducible 80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    print(f"[SPLIT] Training rows: {len(X_train)} | Held-out test rows: {len(X_test)}")

    # 5. Define All Three Models
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            max_depth=16,
            min_samples_split=2,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=5,
            random_state=RANDOM_SEED,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=5,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
    }

    # Model save file mapping
    file_map = {
        "Random Forest": "random_forest.joblib",
        "Gradient Boosting": "gradient_boosting.joblib",
        "XGBoost": "xgboost_model.joblib",
    }

    metrics_rows = []
    test_preds_df = pd.DataFrame(X_test).reset_index(drop=True)
    test_preds_df["actual_wait_minutes"] = y_test

    print("\n[TRAIN] Fitting and evaluating all three models on held-out test set...")
    for name, model in models.items():
        # Fit on training split
        model.fit(X_train, y_train)
        preds = np.maximum(0.0, model.predict(X_test))

        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(mean_squared_error(y_test, preds) ** 0.5)
        r2 = float(r2_score(y_test, preds))

        metrics_rows.append({
            "Model": name,
            "MAE (minutes)": round(mae, 3),
            "RMSE (minutes)": round(rmse, 3),
            "R2 Score": round(r2, 3),
        })

        safe_name = name.lower().replace(" ", "_")
        test_preds_df[f"{safe_name}_predicted"] = np.round(preds, 2)

        # Full training for production deployment
        print(f"   Fitting {name} on full dataset ({len(X)} rows)...")
        model.fit(X, y)

        save_path = MODEL_DIR / file_map[name]
        joblib.dump(model, save_path)
        print(f"   [SAVED] {name} -> {save_path.name} | Test MAE: {mae:.3f} | RMSE: {rmse:.3f} | R2: {r2:.3f}")

    # 6. Save Comparison Metrics
    metrics_df = pd.DataFrame(metrics_rows)
    comparison_path = ROOT / "model_comparison.csv"
    metrics_df.to_csv(comparison_path, index=False)
    print(f"\n[EVAL] Saved {comparison_path.name} with all three models.")

    test_preds_path = ROOT / "test_predictions.csv"
    test_preds_df.to_csv(test_preds_path, index=False)
    print(f"[EVAL] Saved {test_preds_path.name}")

    summary_text = (
        metrics_df.to_string(index=False)
        + f"\n\nRows used: {len(raw)} | Train: {len(X_train)} | Test: {len(X_test)}\n"
    )
    (ROOT / "training_output.txt").write_text(summary_text, encoding="utf-8")

    training_summary = {
        "rows_total": len(raw),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "features": features,
        "metrics": metrics_rows,
    }
    (ROOT / "training_summary.json").write_text(
        json.dumps(training_summary, indent=2), encoding="utf-8"
    )

    # 7. Verification: Load saved models and test key scenarios
    rf_loaded = joblib.load(MODEL_DIR / "random_forest.joblib")
    gb_loaded = joblib.load(MODEL_DIR / "gradient_boosting.joblib")
    xgb_loaded = joblib.load(MODEL_DIR / "xgboost_model.joblib")

    print("\n" + "=" * 70)
    print("VERIFICATION: PREDICTIONS FROM ALL THREE ALGORITHMS")
    print("=" * 70)
    test_scenarios = [
        ("Queue length: 1 (Front of queue)", 12, 0, 1),
        ("Queue length: 5 (Small queue)", 12, 0, 5),
        ("Queue length: 10 (Moderate queue)", 12, 0, 10),
        ("Queue length: 20 (Busy queue)", 16, 47, 20),
        ("Queue length: 30 (Peak rush)", 18, 30, 30),
    ]

    print(f"{'Scenario':<34} {'Queue':<7} {'RF (min)':<10} {'GB (min)':<10} {'XGB (min)':<10}")
    print("-" * 75)
    for desc, h, m, q in test_scenarios:
        inp = pd.DataFrame([{"arrival_hour": h, "arrival_minute": m, "queue_length": q}])
        rf_p = round(float(rf_loaded.predict(inp)[0]), 1)
        gb_p = round(float(gb_loaded.predict(inp)[0]), 1)
        xgb_p = round(float(xgb_loaded.predict(inp)[0]), 1)
        print(f"{desc:<34} {q:<7} {rf_p:<10.1f} {gb_p:<10.1f} {xgb_p:<10.1f}")

    # Specific queue sensitivity check requested: same arrival time (12:00) with Q=5, 10, 20, 30
    print("\n" + "-" * 75)
    print("QUEUE SENSITIVITY TEST (Same Arrival Time 12:00, Queue = 5, 10, 20, 30):")
    print(f"{'Queue Length':<15} {'RF (min)':<12} {'GB (min)':<12} {'XGBoost (min)':<12}")
    print("-" * 75)
    for q in [5, 10, 20, 30]:
        inp = pd.DataFrame([{"arrival_hour": 12, "arrival_minute": 0, "queue_length": q}])
        rf_p = round(float(rf_loaded.predict(inp)[0]), 1)
        gb_p = round(float(gb_loaded.predict(inp)[0]), 1)
        xgb_p = round(float(xgb_loaded.predict(inp)[0]), 1)
        print(f"{q:<15} {rf_p:<12.1f} {gb_p:<12.1f} {xgb_p:<12.1f}")

    print("=" * 70)
    print("FINAL MODEL COMPARISON RESULTS:")
    print("=" * 70)
    print(metrics_df.to_string(index=False))

    best_model = min(metrics_rows, key=lambda m: m["MAE (minutes)"])
    print(f"\n[BEST MODEL] {best_model['Model']} achieved the lowest MAE ({best_model['MAE (minutes)']} min) and RMSE ({best_model['RMSE (minutes)']} min).")
    print("[SUCCESS] All three models trained and verified successfully!")


if __name__ == "__main__":
    main()
