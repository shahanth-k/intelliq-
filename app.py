"""
IntelliQ — Professional Streamlit Dashboard
=============================================
Polished, responsive UI for queue waiting-time prediction and model comparison.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IntelliQ | AI Queue Prediction",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Global */
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: linear-gradient(135deg, #f0f4ff 0%, #fafbff 50%, #f5f0ff 100%); }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px; }

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 50%, #7c3aed 100%);
    border-radius: 20px;
    padding: 3rem 3rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(55,48,163,0.30);
}
.hero-banner::before {
    content: "";
    position: absolute; top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: rgba(255,255,255,0.05);
    border-radius: 50%;
}
.hero-banner::after {
    content: "";
    position: absolute; bottom: -80px; left: 40%;
    width: 200px; height: 200px;
    background: rgba(255,255,255,0.04);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.8rem; font-weight: 800; color: #fff;
    letter-spacing: -0.5px; margin: 0 0 0.5rem; line-height: 1.1;
}
.hero-tag {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    color: rgba(255,255,255,0.9);
    font-size: 0.75rem; font-weight: 600;
    letter-spacing: 1.5px; text-transform: uppercase;
    padding: 4px 12px; border-radius: 20px; margin-bottom: 1rem;
}
.hero-sub {
    font-size: 1.05rem; color: rgba(255,255,255,0.80);
    line-height: 1.6; max-width: 620px;
}
.hero-badges { margin-top: 1.5rem; display: flex; gap: 0.75rem; flex-wrap: wrap; }
.badge {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    color: rgba(255,255,255,0.9);
    font-size: 0.8rem; font-weight: 500;
    padding: 5px 14px; border-radius: 20px;
    backdrop-filter: blur(4px);
}

/* Stat cards */
.stat-grid { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.stat-card {
    flex: 1; min-width: 160px;
    background: #fff;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 2px 20px rgba(0,0,0,0.07);
    border: 1px solid rgba(55,48,163,0.08);
    transition: transform 0.2s, box-shadow 0.2s;
}
.stat-card:hover { transform: translateY(-3px); box-shadow: 0 8px 30px rgba(55,48,163,0.15); }
.stat-label { font-size: 0.75rem; font-weight: 600; color: #6b7280; text-transform: uppercase; letter-spacing: 1px; }
.stat-value { font-size: 2rem; font-weight: 800; color: #1e3a8a; line-height: 1.2; margin: 0.25rem 0; }
.stat-note { font-size: 0.72rem; color: #9ca3af; }

/* Section headers */
.section-header {
    font-size: 1.3rem; font-weight: 700; color: #1f2937;
    margin: 2rem 0 1rem;
    display: flex; align-items: center; gap: 0.5rem;
}
.section-header::after {
    content: ""; flex: 1;
    height: 2px;
    background: linear-gradient(90deg, #3730a3, transparent);
    border-radius: 2px;
    margin-left: 0.75rem;
}

/* Model cards */
.model-card {
    background: #fff;
    border-radius: 16px;
    padding: 1.6rem;
    box-shadow: 0 2px 20px rgba(0,0,0,0.07);
    border: 1px solid #e5e7eb;
    transition: transform 0.2s, box-shadow 0.2s;
    height: 100%;
}
.model-card:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(55,48,163,0.15); }
.model-name { font-size: 1rem; font-weight: 700; color: #1f2937; margin-bottom: 1rem; }
.metric-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.7rem; }
.metric-label { font-size: 0.8rem; color: #6b7280; font-weight: 500; }
.metric-val { font-size: 1.1rem; font-weight: 700; }
.metric-val.good { color: #059669; }
.metric-val.mid { color: #d97706; }
.metric-val.bad { color: #dc2626; }
.metric-bar { height: 6px; border-radius: 3px; background: #e5e7eb; margin-top: 3px; overflow: hidden; }
.metric-fill { height: 100%; border-radius: 3px; background: linear-gradient(90deg, #3730a3, #7c3aed); }

/* Predictor */
.predictor-card {
    background: linear-gradient(135deg, #fff 0%, #f8f7ff 100%);
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 2px 20px rgba(55,48,163,0.10);
    border: 1px solid rgba(55,48,163,0.12);
}
.predict-result {
    background: linear-gradient(135deg, #1e3a8a, #3730a3);
    border-radius: 16px;
    padding: 1.8rem;
    color: #fff;
    text-align: center;
    margin-top: 0.5rem;
}
.predict-result h3 { margin: 0 0 0.5rem; font-size: 0.85rem; font-weight: 600; opacity: 0.8; text-transform: uppercase; letter-spacing: 1px; }
.predict-result .time { font-size: 2.8rem; font-weight: 800; letter-spacing: -1px; }
.predict-result .unit { font-size: 1rem; opacity: 0.7; }

/* Warning banner */
.warn-box {
    background: linear-gradient(135deg, #fffbeb, #fef3c7);
    border: 1px solid #fcd34d;
    border-left: 4px solid #f59e0b;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.85rem;
    color: #78350f;
    margin: 1rem 0;
}

/* Info box */
.info-box {
    background: linear-gradient(135deg, #eff6ff, #dbeafe);
    border: 1px solid #93c5fd;
    border-left: 4px solid #3b82f6;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.85rem;
    color: #1e3a8a;
    margin: 1rem 0;
}

/* Table styling */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #fff;
    border-radius: 12px;
    padding: 4px;
    box-shadow: 0 1px 10px rgba(0,0,0,0.07);
    gap: 0;
    border: 1px solid #e5e7eb;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 0.5rem 1.2rem;
    font-weight: 500;
    color: #6b7280;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1e3a8a, #3730a3) !important;
    color: #fff !important;
}

/* Inputs */
.stSlider, .stNumberInput { padding: 0; }
.stButton > button {
    background: linear-gradient(135deg, #1e3a8a 0%, #7c3aed 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 1.5rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(55,48,163,0.30) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(55,48,163,0.40) !important;
}

/* Download button */
.stDownloadButton > button {
    background: #fff !important;
    color: #3730a3 !important;
    border: 2px solid #3730a3 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    rf = joblib.load(ROOT / "models" / "random_forest.joblib")
    gb = joblib.load(ROOT / "models" / "gradient_boosting.joblib")
    return rf, gb


def color_r2(val: float) -> str:
    if val >= 0.5:
        return "good"
    if val >= 0.1:
        return "mid"
    return "bad"


def color_err(val: float, threshold_good=2.0, threshold_mid=4.0) -> str:
    if val <= threshold_good:
        return "good"
    if val <= threshold_mid:
        return "mid"
    return "bad"


# ── Guard: ensure models are trained ─────────────────────────────────────────
metrics_path = ROOT / "model_comparison.csv"
summary_path = ROOT / "training_summary.json"
models_ready = (
    metrics_path.exists()
    and (ROOT / "models" / "random_forest.joblib").exists()
    and (ROOT / "models" / "gradient_boosting.joblib").exists()
)

if not models_ready:
    st.error("⚠️ Models not trained yet.")
    st.code(
        "pip install -r requirements.txt\npython train_models.py\nstreamlit run app.py",
        language="bash",
    )
    st.stop()

metrics_df = pd.read_csv(metrics_path)
summary = json.loads(summary_path.read_text())
cleaning = summary.get("cleaning_report", {})

rf_model, gb_model = load_models()

# ── Hero Banner ───────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero-banner">
  <div class="hero-tag">🛒 Supermarket Queue Management</div>
  <div class="hero-title">IntelliQ</div>
  <div class="hero-sub">
    Adaptive AI-Based Queue Prediction System — predict customer wait times
    using Random Forest and Gradient Boosting, trained on real supermarket data.
  </div>
  <div class="hero-badges">
    <span class="badge">🌲 Random Forest</span>
    <span class="badge">📈 Gradient Boosting</span>
    <span class="badge">⏱️ Wait Time Prediction</span>
    <span class="badge">🔬 No Target Leakage</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Dataset Stats ─────────────────────────────────────────────────────────────
orig_rows = summary.get("original_rows", "N/A")
sim_rows = summary.get("simulated_rows", "N/A")
total_rows = summary.get("rows_used", "N/A")
train_rows = summary.get("train_rows", "N/A")
test_rows = summary.get("test_rows", "N/A")

st.markdown(
    f"""
<div class="stat-grid">
  <div class="stat-card">
    <div class="stat-label">Total Records</div>
    <div class="stat-value">{total_rows:,}</div>
    <div class="stat-note">Used in pipeline</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Original Records</div>
    <div class="stat-value">{orig_rows}</div>
    <div class="stat-note">Real observations</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Simulated Records</div>
    <div class="stat-value">{sim_rows}</div>
    <div class="stat-note">Synthetic augmentation</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Training Set</div>
    <div class="stat-value">{train_rows:,}</div>
    <div class="stat-note">Incl. simulated rows</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Test Set</div>
    <div class="stat-value">{test_rows}</div>
    <div class="stat-note">Original records only</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

if sim_rows and int(sim_rows) > 0:
    st.markdown(
        f"""
<div class="warn-box">
  ⚠️ <strong>Dataset Notice:</strong> This project uses {orig_rows} real supermarket records
  augmented with {sim_rows} synthetic examples to improve training coverage.
  The test set contains <strong>original records only</strong> for honest evaluation.
  Reported metrics should not be treated as validated real-world accuracy.
</div>
""",
        unsafe_allow_html=True,
    )

# ── Main Tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Model Performance", "🔮 Predict Wait Time", "📂 Dataset Explorer", "🧹 Data Quality"]
)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)

    # Metric guide
    st.markdown(
        """
<div class="info-box">
  📐 <strong>How to read these metrics:</strong>
  <strong>MAE</strong> (Mean Absolute Error) and <strong>RMSE</strong> (Root Mean Squared Error)
  are in minutes — <em>lower is better</em>.
  <strong>R²</strong> measures explained variance — <em>higher is better</em>, with 1.0 being
  a perfect fit and 0 meaning the model is no better than predicting the mean.
  Negative R² indicates the model is worse than the mean baseline on the test set.
</div>
""",
        unsafe_allow_html=True,
    )

    # Side-by-side model cards
    col_rf, col_gb = st.columns(2)

    for col, row in zip([col_rf, col_gb], metrics_df.itertuples()):
        mae = row._2
        rmse = row._3
        r2 = row._4
        # Normalise bars (max reference: MAE/RMSE 10 min, R² ±1)
        mae_pct = min(100, mae / 10 * 100)
        rmse_pct = min(100, rmse / 10 * 100)
        r2_display = max(0, r2 * 100)

        with col:
            icon = "🌲" if "Forest" in row.Model else "📈"
            st.markdown(
                f"""
<div class="model-card">
  <div class="model-name">{icon} {row.Model}</div>
  <div class="metric-row">
    <span class="metric-label">MAE (minutes)</span>
    <span class="metric-val {color_err(mae)}">{mae:.3f}</span>
  </div>
  <div class="metric-bar"><div class="metric-fill" style="width:{mae_pct:.0f}%"></div></div>

  <div class="metric-row" style="margin-top:0.9rem">
    <span class="metric-label">RMSE (minutes)</span>
    <span class="metric-val {color_err(rmse)}">{rmse:.3f}</span>
  </div>
  <div class="metric-bar"><div class="metric-fill" style="width:{rmse_pct:.0f}%"></div></div>

  <div class="metric-row" style="margin-top:0.9rem">
    <span class="metric-label">R² Score</span>
    <span class="metric-val {color_r2(r2)}">{r2:.3f}</span>
  </div>
  <div class="metric-bar"><div class="metric-fill" style="width:{r2_display:.0f}%;background:{'#059669' if r2>0 else '#dc2626'}"></div></div>
</div>
""",
                unsafe_allow_html=True,
            )

    # Bar charts
    st.markdown('<div class="section-header">Visual Comparison</div>', unsafe_allow_html=True)
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        err_chart = metrics_df.set_index("Model")[["MAE (minutes)", "RMSE (minutes)"]]
        st.markdown("**Error Metrics (lower is better)**")
        st.bar_chart(err_chart, height=280, use_container_width=True)

    with chart_col2:
        r2_chart = metrics_df.set_index("Model")[["R2 Score"]]
        st.markdown("**R² Score (higher is better)**")
        st.bar_chart(r2_chart, height=280, use_container_width=True)

    # Full comparison table
    st.markdown('<div class="section-header">Detailed Results Table</div>', unsafe_allow_html=True)
    st.dataframe(
        metrics_df.style.highlight_min(subset=["MAE (minutes)", "RMSE (minutes)"], color="#dcfce7")
                        .highlight_max(subset=["R2 Score"], color="#dcfce7"),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        """
<div class="info-box">
  ℹ️ <strong>Evaluation note:</strong> The test set uses <em>original records only</em> (no
  simulated data), providing a more honest estimate of real-world performance.
  With only ~117 original test records, confidence intervals are wide and metrics
  are indicative rather than definitive.
</div>
""",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Queue Waiting Time Predictor</div>', unsafe_allow_html=True)

    st.markdown(
        """
<div class="info-box">
  🔮 Enter the customer's expected arrival time and current queue length.
  Both models will independently predict the waiting time in minutes.
  Features used: <strong>arrival hour</strong>, <strong>arrival minute</strong>,
  <strong>queue length</strong>. No future information is used.
</div>
""",
        unsafe_allow_html=True,
    )

    pred_left, pred_right = st.columns([1, 1], gap="large")

    with pred_left:
        st.markdown('<div class="predictor-card">', unsafe_allow_html=True)
        st.markdown("#### ⏰ Arrival Time")
        time_col1, time_col2 = st.columns(2)
        with time_col1:
            hour = st.number_input(
                "Hour (0–23)", min_value=0, max_value=23, value=12, step=1, key="pred_hour"
            )
        with time_col2:
            minute = st.number_input(
                "Minute (0–59)", min_value=0, max_value=59, value=0, step=1, key="pred_minute"
            )
        st.markdown("#### 🧍 Queue Status")
        queue_len = st.slider(
            "Current queue length", min_value=0, max_value=57, value=15, step=1, key="pred_queue"
        )
        st.markdown(f"*Queue: {queue_len} customers ahead*")
        st.markdown("")
        predict_btn = st.button("🔮 Predict with Both Models", use_container_width=True, key="predict_btn")
        st.markdown("</div>", unsafe_allow_html=True)

    with pred_right:
        if predict_btn:
            X_input = pd.DataFrame(
                [{"arrival_hour": int(hour), "arrival_minute": int(minute), "queue_length": float(queue_len)}]
            )
            rf_pred = max(0.0, float(rf_model.predict(X_input)[0]))
            gb_pred = max(0.0, float(gb_model.predict(X_input)[0]))
            avg_pred = (rf_pred + gb_pred) / 2.0

            st.markdown(
                f"""
<div class="predict-result">
  <h3>🌲 Random Forest</h3>
  <div class="time">{rf_pred:.1f}</div>
  <div class="unit">minutes estimated wait</div>
</div>
""",
                unsafe_allow_html=True,
            )
            st.markdown("")
            st.markdown(
                f"""
<div class="predict-result" style="background: linear-gradient(135deg, #7c3aed, #be185d)">
  <h3>📈 Gradient Boosting</h3>
  <div class="time">{gb_pred:.1f}</div>
  <div class="unit">minutes estimated wait</div>
</div>
""",
                unsafe_allow_html=True,
            )
            st.markdown("")
            st.info(
                f"📊 **Ensemble average**: {avg_pred:.1f} min  |  "
                f"**Difference between models**: {abs(rf_pred - gb_pred):.1f} min"
            )
            # Agreement indicator
            diff = abs(rf_pred - gb_pred)
            if diff < 1.0:
                st.success("✅ Both models are in strong agreement.")
            elif diff < 3.0:
                st.warning("⚠️ Models show moderate disagreement. Consider the average estimate.")
            else:
                st.error("❌ Models diverge significantly. Use caution interpreting these results.")
        else:
            st.markdown(
                """
<div style="background:#fff;border-radius:20px;padding:3rem;text-align:center;
            box-shadow:0 2px 20px rgba(0,0,0,0.07);border:1px solid #e5e7eb;margin-top:1rem">
  <div style="font-size:3rem">🔮</div>
  <div style="font-size:1.1rem;font-weight:600;color:#374151;margin:1rem 0 0.5rem">
    Ready to Predict
  </div>
  <div style="color:#6b7280;font-size:0.9rem">
    Set arrival time and queue length,<br>then click the Predict button.
  </div>
</div>
""",
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DATASET EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Dataset Explorer</div>', unsafe_allow_html=True)

    data = pd.read_csv(ROOT / "queue_data.csv")

    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.metric("Total Rows", f"{len(data):,}")
    info_col2.metric("Columns", len(data.columns))
    info_col3.metric("Columns", ", ".join(data.columns[:3]) + "...")

    st.markdown(
        """
<div class="info-box">
  📋 <strong>Column definitions:</strong><br>
  • <code>arrival_time</code> — when the customer arrived at the queue<br>
  • <code>start_time</code> — when service began (customer reached the counter)<br>
  • <code>finish_time</code> — when service ended<br>
  • <code>wait_time</code> — <strong>start_time − arrival_time</strong> (in minutes) — the prediction target<br>
  • <code>queue_length</code> — number of people in queue when customer arrived
</div>
""",
        unsafe_allow_html=True,
    )

    # Preview rows
    rows_to_show = st.slider("Rows to preview", min_value=5, max_value=100, value=20, step=5)
    st.dataframe(data.head(rows_to_show), use_container_width=True, hide_index=True)

    # Stats
    st.markdown('<div class="section-header">Summary Statistics</div>', unsafe_allow_html=True)
    numeric_cols = data.select_dtypes(include="number")
    st.dataframe(numeric_cols.describe().round(2), use_container_width=True)

    # Wait-time histogram via Streamlit
    st.markdown('<div class="section-header">Wait Time Distribution</div>', unsafe_allow_html=True)
    hist_data = data["wait_time"].dropna()
    hist_counts, hist_edges = np.histogram(hist_data, bins=30)
    hist_df = pd.DataFrame({"Wait Time (min)": hist_edges[:-1], "Count": hist_counts})
    st.bar_chart(hist_df.set_index("Wait Time (min)"), height=280)

    # Download button
    st.markdown('<div class="section-header">Download Cleaned Dataset</div>', unsafe_allow_html=True)
    csv_bytes = data.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download queue_data.csv",
        data=csv_bytes,
        file_name="intelliq_queue_data.csv",
        mime="text/csv",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DATA QUALITY / CLEANING REPORT
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Data Cleaning Report</div>', unsafe_allow_html=True)

    st.markdown(
        """
<div class="info-box">
  🧹 <strong>Cleaning pipeline summary:</strong> Each row was validated by parsing timestamps,
  deriving <code>wait_time = start_time − arrival_time</code>, and filtering out negative
  or implausible values. Simulated rows were generated from observed statistical patterns
  and added to training only (not to the test set).
</div>
""",
        unsafe_allow_html=True,
    )

    # Display cleaning stats as a styled table
    report_rows = []
    label_map = {
        "total_raw_rows": ("Total raw rows loaded", "ℹ️"),
        "invalid_arrival_ts": ("Invalid arrival timestamps", "❌"),
        "invalid_start_ts": ("Invalid start timestamps", "❌"),
        "missing_finish_ts": ("Missing finish timestamps", "⚠️"),
        "missing_queue_length": ("Missing queue length values", "⚠️"),
        "missing_wait_time": ("Missing wait_time values", "⚠️"),
        "duplicate_rows": ("Duplicate rows", "⚠️"),
        "negative_derived_wait": ("Negative derived wait times", "❌"),
        "excessive_wait_gt_180min": ("Wait time > 180 min (dropped)", "❌"),
        "wait_time_mismatch_gt_1min": ("Wait-time mismatch > 1 min", "🔶"),
        "rows_retained": ("Rows retained after cleaning", "✅"),
        "rows_dropped": ("Rows dropped", "❌"),
        "original_records_retained": ("Original records in cleaned set", "✅"),
        "simulated_records_retained": ("Simulated records in cleaned set", "🟡"),
    }
    for key, (label, icon) in label_map.items():
        val = cleaning.get(key, "N/A")
        report_rows.append({"Check": f"{icon} {label}", "Count": val})

    report_df = pd.DataFrame(report_rows)
    st.dataframe(report_df, use_container_width=True, hide_index=True)

    # Evaluation strategy note
    st.markdown('<div class="section-header">Evaluation Strategy</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
| Parameter | Value |
|-----------|-------|
| Total records used | {total_rows:,} |
| Original records | {orig_rows} |
| Simulated records (train only) | {sim_rows} |
| Train set size | {train_rows:,} |
| Test set size | {test_rows} |
| Test set composition | Original records only |
| Train/test split | 80% / 20% of originals |
| Random seed | 42 |
| Target definition | `wait_time = start_time − arrival_time` (minutes) |
| Features | arrival_hour, arrival_minute, queue_length |
| Leakage check | start_time and finish_time NOT used as features ✅ |
"""
    )

    eval_note = summary.get("evaluation_note", "")
    if eval_note:
        st.markdown(
            f"""
<div class="warn-box">
  ⚠️ <strong>Evaluation note:</strong> {eval_note}
</div>
""",
            unsafe_allow_html=True,
        )

    # Test predictions preview
    preds_path = ROOT / "test_predictions.csv"
    if preds_path.exists():
        st.markdown('<div class="section-header">Test Set Predictions (sample)</div>', unsafe_allow_html=True)
        preds = pd.read_csv(preds_path)
        st.dataframe(preds.head(20).round(2), use_container_width=True, hide_index=True)
        preds_bytes = preds.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download test_predictions.csv",
            data=preds_bytes,
            file_name="intelliq_test_predictions.csv",
            mime="text/csv",
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div style="text-align:center;padding:2rem 0 1rem;color:#9ca3af;font-size:0.8rem;">
  IntelliQ · AI-Based Queue Prediction &amp; Management System ·
  <strong style="color:#6b7280;">Random Forest, Gradient Boosting &amp; XGBoost</strong> ·
  No target leakage · Original data evaluation
</div>
""",
    unsafe_allow_html=True,
)
