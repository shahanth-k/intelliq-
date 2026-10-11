"""
IntelliQ — Google Colab Runner & Demonstration Script
======================================================
This standalone script is designed to run directly in Google Colab.
It trains Random Forest, Gradient Boosting, and XGBoost, evaluates them on
held-out test data, and displays comprehensive console outputs and visualizations.

To run in Google Colab:
    Option A (Recommended — 2 lines in Colab):
        !git clone https://github.com/shahanth-k/intelliq-.git
        %cd intelliq-
        !python intelliq_colab_demo.py

    Option B:
        Upload this file and queue_data.csv to Colab and run:
        !python intelliq_colab_demo.py
"""

import sys
import subprocess
from pathlib import Path

# Ensure dependencies are available in Colab environment
required_packages = ["xgboost", "joblib", "scikit-learn", "pandas", "numpy", "matplotlib"]
for pkg in required_packages:
    try:
        __import__(pkg)
    except ImportError:
        print(f"[SETUP] Installing missing package: {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "queue_data.csv"

# If queue_data.csv is not present in local folder, attempt download from GitHub
if not DATA_PATH.exists():
    github_raw = "https://raw.githubusercontent.com/shahanth-k/intelliq-/main/queue_data.csv"
    print(f"[DATA] queue_data.csv not found locally. Downloading from GitHub...")
    import urllib.request
    urllib.request.urlretrieve(github_raw, DATA_PATH)
    print("[DATA] Download complete.")


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
    print("=" * 75)
    print("         INTELLIQ — QUEUE WAITING TIME ML ENGINE (GOOGLE COLAB)        ")
    print("      Algorithms: Random Forest | Gradient Boosting | XGBoost Regressor")
    print("=" * 75)

    # 1. Load Dataset
    raw = pd.read_csv(DATA_PATH)
    print(f"\n[1] DATASET INSPECTION")
    print(f"    • Total records loaded: {len(raw):,} rows")
    print(f"    • Columns: {list(raw.columns)}")
    print("\nFirst 5 rows preview:")
    print(raw.head().to_string(index=False))

    # 2. Target & Timestamp Verification
    print("\n" + "=" * 75)
    print("[2] DATA PREPARATION & TIMESTAMP VERIFICATION")
    arrivals = raw["arrival_time"].map(parse_clock)
    starts = raw["start_time"].map(parse_clock)
    
    # Calculate difference in minutes with midnight rollover modulo 1440
    calc_wait = ((starts - arrivals).dt.total_seconds() / 60.0) % 1440
    raw_wait = raw["wait_time"]
    discrepancy = (raw_wait - calc_wait).abs()

    exact_matches = int((discrepancy < 0.001).sum())
    minor_fractional = int(((discrepancy >= 0.001) & (discrepancy <= 1.0)).sum())
    large_mismatches = int((discrepancy > 1.0).sum())

    print(f"    • Target definition: wait_time = start_time - arrival_time (minutes)")
    print(f"    • Exact timestamp matches: {exact_matches}")
    print(f"    • Sub-minute rounding differences (<= 1.0 min): {minor_fractional}")
    print(f"      (Explanation: 'start_time' is recorded to integer minutes; wait_time retains decimals)")
    print(f"    • Large mismatches (> 1.0 min): {large_mismatches}")
    print(f"    • Midnight rollover: Properly resolved using modulo 1440 minutes")

    # Features: arrival_hour, arrival_minute, queue_length (no future leakage)
    X = pd.DataFrame({
        "arrival_hour": arrivals.dt.hour.values,
        "arrival_minute": arrivals.dt.minute.values,
        "queue_length": raw["queue_length"].values,
    })
    y = raw["wait_time"].values

    # 3. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n[3] TRAIN / TEST SPLIT (80% Train, 20% Test)")
    print(f"    • Training samples: {len(X_train):,} rows")
    print(f"    • Testing samples:  {len(X_test):,} rows")

    # 4. Train All Three Models
    print("\n" + "=" * 75)
    print("[4] TRAINING AND EVALUATING ALL THREE REGRESSION ALGORITHMS")
    print("=" * 75)

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=150, max_depth=16, min_samples_split=2, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42, n_jobs=-1
        ),
    }

    metrics_list = []
    trained_models = {}

    for name, model in models.items():
        print(f"--> Training {name} Regressor...")
        model.fit(X_train, y_train)
        preds = np.maximum(0.0, model.predict(X_test))

        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(mean_squared_error(y_test, preds) ** 0.5)
        r2 = float(r2_score(y_test, preds))

        metrics_list.append({
            "Algorithm": name,
            "MAE (min)": round(mae, 3),
            "RMSE (min)": round(rmse, 3),
            "R² Score": round(r2, 3),
        })

        # Retrain on 100% of data for live predictions
        model.fit(X, y)
        trained_models[name] = model

    # 5. Model Comparison Table
    print("\n" + "=" * 75)
    print("[5] MODEL COMPARISON EVALUATION RESULTS (HELD-OUT TEST SET)")
    print("=" * 75)
    metrics_df = pd.DataFrame(metrics_list)
    print(metrics_df.to_string(index=False))

    best = min(metrics_list, key=lambda m: m["MAE (min)"])
    print(f"\n>>> BEST PERFORMING MODEL: {best['Algorithm']} (Lowest MAE: {best['MAE (min)']} min, RMSE: {best['RMSE (min)']} min)")

    # 6. Scenario Test & Queue Length Sensitivity
    print("\n" + "=" * 75)
    print("[6] SENSITIVITY TEST: PREDICTIONS ACROSS QUEUE LENGTHS (Arrival: 12:00)")
    print("=" * 75)
    print(f"{'Queue Length':<15} {'People Ahead':<15} {'RF (min)':<12} {'GB (min)':<12} {'XGBoost (min)':<12}")
    print("-" * 75)
    for q in [1, 5, 10, 20, 30]:
        inp = pd.DataFrame([{"arrival_hour": 12, "arrival_minute": 0, "queue_length": q}])
        rf_p = round(float(trained_models["Random Forest"].predict(inp)[0]), 1)
        gb_p = round(float(trained_models["Gradient Boosting"].predict(inp)[0]), 1)
        xgb_p = round(float(trained_models["XGBoost"].predict(inp)[0]), 1)
        ahead = max(0, q - 1)
        print(f"{q:<15} {ahead:<15} {rf_p:<12.1f} {gb_p:<12.1f} {xgb_p:<12.1f}")

    # 7. Verification Mockup Scenarios
    print("\n" + "=" * 75)
    print("[7] KEY VERIFICATION SCENARIOS")
    print("=" * 75)
    scenarios = [
        ("Front of Queue (12:00, Q=1)", 12, 0, 1),
        ("Small Queue Mockup 1 (12:00, Q=5)", 12, 0, 5),
        ("Moderate Queue (12:00, Q=10)", 12, 0, 10),
        ("Busy Queue Mockup 2 (16:47, Q=20)", 16, 47, 20),
        ("Peak Rush Queue (18:30, Q=30)", 18, 30, 30),
    ]
    for desc, h, m, q in scenarios:
        inp = pd.DataFrame([{"arrival_hour": h, "arrival_minute": m, "queue_length": q}])
        rf_val = round(float(trained_models["Random Forest"].predict(inp)[0]), 1)
        gb_val = round(float(trained_models["Gradient Boosting"].predict(inp)[0]), 1)
        xgb_val = round(float(trained_models["XGBoost"].predict(inp)[0]), 1)
        print(f"• {desc:<36} -> RF: {rf_val:4.1f}m | GB: {gb_val:4.1f}m | XGB: {xgb_val:4.1f}m")

    # 8. Visual Plots (Generates plot if running in notebook or GUI environment)
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Subplot 1: Error metrics comparison
        algos = [m["Algorithm"] for m in metrics_list]
        maes = [m["MAE (min)"] for m in metrics_list]
        rmses = [m["RMSE (min)"] for m in metrics_list]
        x_idx = np.arange(len(algos))
        w = 0.35

        ax1.bar(x_idx - w/2, maes, w, label="MAE (minutes)", color="#3b82f6")
        ax1.bar(x_idx + w/2, rmses, w, label="RMSE (minutes)", color="#ef4444")
        ax1.set_title("Model Error Comparison (Lower is Better)")
        ax1.set_xticks(x_idx)
        ax1.set_xticklabels(algos)
        ax1.set_ylabel("Minutes")
        ax1.legend()
        ax1.grid(axis="y", linestyle="--", alpha=0.5)

        # Subplot 2: Queue sensitivity curve
        test_q = np.arange(1, 41)
        df_curve = pd.DataFrame({
            "arrival_hour": 12,
            "arrival_minute": 0,
            "queue_length": test_q
        })
        ax2.plot(test_q, trained_models["Random Forest"].predict(df_curve), label="Random Forest", color="#10b981", linewidth=2)
        ax2.plot(test_q, trained_models["Gradient Boosting"].predict(df_curve), label="Gradient Boosting", color="#f59e0b", linewidth=2)
        ax2.plot(test_q, trained_models["XGBoost"].predict(df_curve), label="XGBoost", color="#2563eb", linewidth=2, linestyle="--")
        ax2.set_title("Predicted Waiting Time vs. Queue Length (Arrival: 12:00)")
        ax2.set_xlabel("Queue Length (people)")
        ax2.set_ylabel("Wait Time (minutes)")
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plot_path = ROOT / "colab_model_comparison_plot.png"
        plt.savefig(plot_path, dpi=120)
        print(f"\n[PLOT] Comparison charts generated and saved to: {plot_path.name}")
        
        # Display inline if in Google Colab / IPython, otherwise close plot
        if "google.colab" in sys.modules or "IPython" in sys.modules:
            plt.show()
        else:
            plt.close()
    except Exception as e:
        print(f"[PLOT] Plot rendering skipped: {e}")

    print("\n" + "=" * 75)
    print("SUCCESS: Pipeline executed completely in Colab!")
    print("=" * 75)


if __name__ == "__main__":
    main()
