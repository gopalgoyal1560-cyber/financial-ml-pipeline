"""
Financial ML Pipeline — Prototype Dashboard
=============================================
Drop this file into the ROOT of your cloned repo
(financial-ml-pipeline/app.py) so it can find:
    data/processed/daily_latest.csv
    data/processed/merged_imb_dataset2.csv
    log/daily_ingestion.log
    pipeline_rasso.pkl   (optional, only used to show feature list)

Run with:
    pip install streamlit pandas requests plotly numpy
    streamlit run app.py
"""

import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Financial ML Pipeline",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "https://financial-ml-pipeline-up86.onrender.com/Post_values"
DAILY_CSV = Path("data/processed/daily_latest.csv")
TRAIN_CSV = Path("data/processed/merged_imb_dataset2.csv")
LOG_FILE = Path("log/daily_ingestion.log")

# ----------------------------------------------------------------------------
# CUSTOM CSS
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }

        .main {
            background-color: #0e1117;
        }

        /* Hero header */
        .hero {
            background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
            padding: 2.2rem 2rem;
            border-radius: 18px;
            margin-bottom: 1.6rem;
            border: 1px solid rgba(255,255,255,0.06);
        }
        .hero h1 {
            color: #ffffff;
            font-size: 2.1rem;
            font-weight: 800;
            margin-bottom: 0.3rem;
        }
        .hero p {
            color: #b8c6d2;
            font-size: 1.02rem;
            margin: 0;
        }
        .status-pill {
            display: inline-block;
            padding: 0.25rem 0.9rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            margin-top: 0.9rem;
            background: rgba(255, 196, 0, 0.15);
            color: #ffc400;
            border: 1px solid rgba(255,196,0,0.35);
        }

        /* Metric cards */
        div[data-testid="stMetric"] {
            background: linear-gradient(145deg, #161b22, #1c232c);
            border: 1px solid rgba(255,255,255,0.06);
            padding: 1rem 1.1rem;
            border-radius: 14px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.25);
        }
        div[data-testid="stMetricLabel"] { color: #93a1b0 !important; }

        /* Section card */
        .card {
            background: #141a21;
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 16px;
            padding: 1.4rem 1.6rem;
            margin-bottom: 1.3rem;
        }
        .card h3 { margin-top: 0; color: #eaf1f7; }
        .card p, .card li { color: #aebac6; }

        /* Pipeline flow diagram */
        .flow-wrap {
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin: 1rem 0;
        }
        .flow-box {
            background: #1b2430;
            border: 1px solid #2c3a4a;
            border-radius: 12px;
            padding: 0.7rem 1rem;
            color: #dbe6ef;
            font-size: 0.85rem;
            font-weight: 600;
            text-align: center;
            min-width: 120px;
        }
        .flow-arrow {
            color: #4fd1c5;
            font-size: 1.3rem;
            font-weight: 700;
        }

        /* Badges */
        .badge-up {
            background: rgba(46, 204, 113, 0.15);
            color: #2ecc71;
            border: 1px solid rgba(46,204,113,0.4);
            padding: 0.5rem 1.1rem;
            border-radius: 10px;
            font-weight: 700;
            display: inline-block;
        }
        .badge-down {
            background: rgba(231, 76, 60, 0.15);
            color: #e74c3c;
            border: 1px solid rgba(231,76,60,0.4);
            padding: 0.5rem 1.1rem;
            border-radius: 10px;
            font-weight: 700;
            display: inline-block;
        }
        .footnote { color: #6b7885; font-size: 0.78rem; margin-top: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# STATIC DATA — extracted directly from model1_proto.ipynb run outputs
# (final held-out test set, "touched once", so all rows are comparable)
# ----------------------------------------------------------------------------
MODEL_RESULTS = pd.DataFrame([
    {"Model": "Baseline — Always 0",                         "MAE": 0.011985, "R2": -0.001892, "Type": "Baseline"},
    {"Model": "Baseline — Always Mean",                       "MAE": 0.011968, "R2": -0.001186, "Type": "Baseline"},
    {"Model": "Baseline — Copy Last Value",                   "MAE": 0.017140, "R2": -0.871054, "Type": "Baseline"},
    {"Model": "Pipe1 — Linear Reg (median impute)",           "MAE": 0.022863, "R2": -2.123576, "Type": "Trained"},
    {"Model": "Pipe2 — Linear Reg (impute + scale)",          "MAE": 0.022863, "R2": -2.123576, "Type": "Trained"},
    {"Model": "Pipe3 — Linear Reg (drop corr. features)",     "MAE": 0.018929, "R2": -3.217026, "Type": "Trained"},
    {"Model": "Pipe4 — Linear Reg (PCA)",                     "MAE": 0.019317, "R2": -2.987670, "Type": "Trained"},
    {"Model": "Pipe5 — Linear Reg (drop price-level cols)",   "MAE": 0.014605, "R2": -0.547534, "Type": "Trained"},
    {"Model": "Ridge (RidgeCV, alpha=1000)",                  "MAE": 0.017502, "R2": -0.618449, "Type": "Trained"},
    {"Model": "Lasso (LassoCV, alpha=0.001)  ★ SELECTED",     "MAE": 0.012098, "R2": -0.007286, "Type": "Selected"},
    {"Model": "Random Forest (full features)",                "MAE": 0.012969, "R2": -0.125011, "Type": "Trained"},
    {"Model": "Random Forest (PCA)",                          "MAE": 0.012936, "R2": -0.108755, "Type": "Trained"},
])

LASSO_KEPT = pd.DataFrame([{"Feature": "adjusted", "Coefficient": -0.000131}])
LASSO_DROPPED = [
    "close", "high", "low", "open", "volume", "SMA_10", "SMA_20", "SMA_50",
    "EMA_10", "EMA_20", "EMA_50", "RSI_14", "MACD", "MACD_signal", "MACD_hist",
    "BB_mid_20", "BB_upper_20", "BB_lower_20", "BB_pctB_20", "ATR", "OBV",
    "volatility_10", "volatility_20", "momentum_10", "roc_10", "momentum_20",
    "roc_20", "Value", "wti_value", "gold_price",
]

RF_IMPORTANCE = pd.DataFrame([
    {"Feature": "MACD_hist",      "Importance": 0.111334},
    {"Feature": "volume",         "Importance": 0.088844},
    {"Feature": "close",          "Importance": 0.074575},
    {"Feature": "adjusted",       "Importance": 0.069257},
    {"Feature": "volatility_10",  "Importance": 0.066571},
    {"Feature": "low",            "Importance": 0.058425},
    {"Feature": "Value",          "Importance": 0.046797},
    {"Feature": "high",           "Importance": 0.040167},
    {"Feature": "OBV",            "Importance": 0.035233},
    {"Feature": "volatility_20",  "Importance": 0.033419},
])

TRAIN_TEST_DESCRIBE = pd.DataFrame({
    "count": [2676, 1147],
    "mean":  [0.000175, 0.000842],
    "std":   [0.014481, 0.019374],
    "min":   [-0.128507, -0.252076],
    "25%":   [-0.006230, -0.007521],
    "50%":   [0.000330, 0.001247],
    "75%":   [0.006994, 0.009111],
    "max":   [0.113010, 0.129642],
}, index=["Y_train (return_pct)", "Y_test (return_pct)"])

PIPELINE_STEPS = [
    "Yahoo Finance\n(IBM OHLCV)",
    "Alpha Vantage\n(Treasury, WTI, Gold)",
    "Schema\nValidation",
    "Feature\nEngineering",
    "Merge &\nForward-fill",
    "daily_latest.csv",
    "FastAPI\n/Post_values",
    "Lasso Model\nPrediction",
]

DATA_SOURCES = pd.DataFrame([
    {"Source": "IBM daily OHLCV", "Provider": "Yahoo Finance (yfinance)", "Fields": "open, high, low, close, adjusted close, volume"},
    {"Source": "30Y Treasury yield", "Provider": "Alpha Vantage", "Fields": "daily yield"},
    {"Source": "WTI crude oil", "Provider": "Alpha Vantage", "Fields": "daily price"},
    {"Source": "Gold price", "Provider": "Alpha Vantage", "Fields": "daily price"},
])


# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
@st.cache_data(ttl=300)
def load_daily_csv():
    if DAILY_CSV.exists():
        df = pd.read_csv(DAILY_CSV)
        df["Date"] = pd.to_datetime(df["Date"])
        return df
    return None


@st.cache_data(ttl=300)
def load_training_csv():
    if TRAIN_CSV.exists():
        df = pd.read_csv(TRAIN_CSV)
        return df
    return None


@st.cache_data(ttl=300)
def parse_prediction_log():
    """Parse log/daily_ingestion.log for (data_date, predicted_return) pairs."""
    if not LOG_FILE.exists():
        return pd.DataFrame(columns=["run_timestamp", "data_date", "predicted_return"])

    text = LOG_FILE.read_text(errors="ignore")
    lines = text.splitlines()

    rows = []
    last_data_date = None
    for line in lines:
        m_saved = re.search(r"Saved latest row \| file : .* \| date : ([\d-]+)", line)
        if m_saved:
            last_data_date = m_saved.group(1)
            continue

        m_pred = re.search(
            r"^([\d\-]+ [\d:,]+).*Prediction request SUCCESS.*predictions': \[([-\d.eE]+)\]",
            line,
        )
        if m_pred:
            ts_str, pred_str = m_pred.groups()
            try:
                ts = datetime.strptime(ts_str.split(",")[0], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                ts = None
            rows.append({
                "run_timestamp": ts,
                "data_date": last_data_date,
                "predicted_return": float(pred_str),
            })

    return pd.DataFrame(rows)


def call_live_api(payload: dict):
    try:
        resp = requests.post(API_URL, json=payload, timeout=(5, 30))
        resp.raise_for_status()
        return resp.json(), None
    except Exception as e:
        return None, str(e)


# ----------------------------------------------------------------------------
# HERO HEADER
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>📈 Financial ML Pipeline — Dashboard</h1>
        <p>End-to-end view of the ingestion → validation → feature engineering → model → prediction API pipeline for IBM daily returns.</p>
        <span class="status-pill">⚠ EXPERIMENTAL MODEL — SEE MODEL COMPARISON TAB</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div style="
        background: rgba(231, 76, 60, 0.12);
        border: 1px solid rgba(231, 76, 60, 0.4);
        border-radius: 12px;
        padding: 1rem 1.3rem;
        margin-bottom: 1.4rem;
        color: #f5b7b1;
        font-size: 0.9rem;
        line-height: 1.5;
    ">
        <strong>⚠️ Disclaimer:</strong> This is a personal portfolio project built to demonstrate an
        end-to-end ML pipeline (data ingestion, feature engineering, model training, and deployment).
        It is <strong>not financial advice</strong> and comes with <strong>no guarantee of accuracy,
        reliability, or performance</strong>. The model's predictions are experimental and, as shown in
        the Model Comparison tab, do not reliably outperform a naive baseline. Do <strong>not</strong>
        use this app, its outputs, or its predictions to make investment decisions or to analyze real
        financial markets.
    </div>
    """,
    unsafe_allow_html=True,
)
# ----------------------------------------------------------------------------
# TOP-LEVEL KPIs
# ----------------------------------------------------------------------------
daily_df = load_daily_csv()
train_df = load_training_csv()

k1, k2, k3, k4 = st.columns(4)
if daily_df is not None and len(daily_df) > 0:
    latest = daily_df.iloc[-1]
    k1.metric("Latest Date", latest["Date"].strftime("%Y-%m-%d"))
    k2.metric("Latest Close", f"${latest['close']:.2f}")
    k3.metric("Latest RSI (14)", f"{latest['RSI_14']:.1f}")
    k4.metric("Last Return %", f"{latest['return_pct']*100:.2f}%")
else:
    k1.metric("Latest Date", "—")
    k2.metric("Latest Close", "—")
    k3.metric("Latest RSI (14)", "—")
    k4.metric("Last Return %", "—")
    st.warning(f"Couldn't find `{DAILY_CSV}` — place this file at the root of the cloned repo so relative paths resolve.")

st.write("")

# ----------------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------------
tab_overview, tab_live, tab_history, tab_models, tab_features, tab_explorer = st.tabs(
    ["🗺️ Overview", "🔮 Live Prediction", "📅 Prediction History",
     "🧪 Model Comparison", "🧬 Feature Analysis", "🗂️ Dataset Explorer"]
)

# ---------------- OVERVIEW ----------------
with tab_overview:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Pipeline Flow")
    flow_html = '<div class="flow-wrap">'
    for i, step in enumerate(PIPELINE_STEPS):
        flow_html += f'<div class="flow-box">{step}</div>'
        if i != len(PIPELINE_STEPS) - 1:
            flow_html += '<div class="flow-arrow">➜</div>'
    flow_html += "</div>"
    st.markdown(flow_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Data Sources")
        st.dataframe(DATA_SOURCES, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Automation")
        st.markdown(
            """
            - **daily_fetch.yml** — weekday cron, fetches + validates + engineers features, commits new row
            - **response.yml** — triggers after fetch, posts latest row to the deployed model API, logs response
            - Schema drift check (`genson` + `deepdiff`) halts the pipeline on breaking upstream changes
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Roadmap")
    st.markdown(
        """
        - [ ] Expand beyond a single equity (IBM) to test whether weak signal is asset-specific
        - [ ] Formal model evaluation framework / experiment tracking
        - [ ] Production model selection criteria
        - [ ] Dockerize the API service
        - [ ] Monitoring for the deployed prediction endpoint and data pipeline
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- LIVE PREDICTION ----------------
with tab_live:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Get a Live Prediction from the Deployed API")
    st.caption(f"POSTs the most recent row in `daily_latest.csv` to `{API_URL}`")

    if daily_df is None or len(daily_df) == 0:
        st.error("No daily_latest.csv found — can't build a request payload.")
    else:
        latest_row = daily_df.iloc[-1:].drop(columns=["Date"])
        with st.expander("📋 View input feature row being sent"):
            st.dataframe(latest_row.T.rename(columns={latest_row.index[0]: "value"}), use_container_width=True)

        if st.button("🚀 Request Prediction", type="primary"):
            payload = latest_row.to_dict(orient="records")[0]
            with st.spinner("Calling deployed model API (may take up to ~30s if the service is cold)..."):
                result, error = call_live_api(payload)

            if error:
                st.error(f"Request failed: {error}")
            else:
                pred = result["predictions"][0]
                direction = "UP" if pred >= 0 else "DOWN"
                badge_class = "badge-up" if pred >= 0 else "badge-down"
                arrow = "▲" if pred >= 0 else "▼"
                st.markdown(
                    f'<span class="{badge_class}">{arrow} Predicted next-day return: {pred*100:.4f}%  ({direction})</span>',
                    unsafe_allow_html=True,
                )
                st.json(result)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICTION HISTORY ----------------
with tab_history:
    log_df = parse_prediction_log()
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Recent Logged Predictions")
    st.caption("Parsed directly from `log/daily_ingestion.log`")

    if log_df.empty:
        st.info("No prediction log entries found yet.")
    else:
        log_df_sorted = log_df.sort_values("run_timestamp").tail(5).reset_index(drop=True)
        display_df = log_df_sorted.copy()
        display_df["predicted_return_%"] = (display_df["predicted_return"] * 100).round(4)

        # try to attach actual realized return if we can match on date
        if daily_df is not None:
            merged = display_df.merge(
                daily_df[["Date", "return_pct"]].assign(Date=daily_df["Date"].dt.strftime("%Y-%m-%d")),
                left_on="data_date", right_on="Date", how="left"
            )
            merged["actual_return_%"] = (merged["return_pct"] * 100).round(4)
        else:
            merged = display_df
            merged["actual_return_%"] = np.nan

        st.dataframe(
            merged[["run_timestamp", "data_date", "predicted_return_%", "actual_return_%"]],
            use_container_width=True, hide_index=True
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=merged["data_date"], y=merged["predicted_return_%"],
            mode="lines+markers", name="Predicted %", line=dict(color="#4fd1c5", width=3)
        ))
        if merged["actual_return_%"].notna().any():
            fig.add_trace(go.Scatter(
                x=merged["data_date"], y=merged["actual_return_%"],
                mode="lines+markers", name="Actual %", line=dict(color="#f39c12", width=3, dash="dash")
            ))
        fig.update_layout(
            title="Predicted vs Actual Return — Last 5 Logged Runs",
            template="plotly_dark", height=420,
            xaxis_title="Data Date", yaxis_title="Return (%)",
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- MODEL COMPARISON ----------------
with tab_models:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Algorithms Tried (from `model1_proto.ipynb`)")
    st.caption("All metrics below are from the same held-out test set, evaluated once — directly comparable.")

    st.dataframe(
        MODEL_RESULTS.style.format({"MAE": "{:.5f}", "R2": "{:.4f}"}),
        use_container_width=True, hide_index=True
    )

    color_map = {"Baseline": "#7f8c8d", "Trained": "#5b8def", "Selected": "#2ecc71"}
    mae_fig = px.bar(
        MODEL_RESULTS.sort_values("MAE"), x="MAE", y="Model", color="Type",
        orientation="h", color_discrete_map=color_map,
        title="Mean Absolute Error — lower is better (baseline 'Always 0' = 0.01199)"
    )
    mae_fig.update_layout(template="plotly_dark", height=480, yaxis={'categoryorder': 'total descending'})
    st.plotly_chart(mae_fig, use_container_width=True)

    r2_fig = px.bar(
        MODEL_RESULTS.sort_values("R2"), x="R2", y="Model", color="Type",
        orientation="h", color_discrete_map=color_map,
        title="R² Score — closer to 0 (or positive) is better; all models here are ≤ 0"
    )
    r2_fig.update_layout(template="plotly_dark", height=480, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(r2_fig, use_container_width=True)

    st.info(
        "**Honest takeaway from the notebook:** none of the trained pipelines clearly beat the naive "
        "'always predict 0' baseline. Lasso came closest (MAE 0.01210 vs baseline 0.01199) by zeroing out "
        "30 of 31 features — read as evidence of weak/noisy signal in these engineered features for a single equity."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Train vs Test Target Distribution — the 'regime shift' check")
    st.dataframe(TRAIN_TEST_DESCRIBE.style.format("{:.5f}"), use_container_width=True)
    st.caption(
        "Std deviation and range are noticeably larger in the test split than in train — consistent with the "
        "notebook's hypothesis that the test period contained a sharper regime shift than any model trained here could handle."
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FEATURE ANALYSIS ----------------
with tab_features:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Lasso Feature Selection")
        st.caption(f"Kept: {len(LASSO_KEPT)} feature — dropped to zero: {len(LASSO_DROPPED)} features")
        st.dataframe(LASSO_KEPT, use_container_width=True, hide_index=True)
        with st.expander(f"Show all {len(LASSO_DROPPED)} dropped features"):
            st.write(", ".join(LASSO_DROPPED))
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Random Forest — Top 10 Feature Importances")
        fig = px.bar(
            RF_IMPORTANCE.sort_values("Importance"), x="Importance", y="Feature",
            orientation="h", color="Importance", color_continuous_scale="Tealgrn",
        )
        fig.update_layout(template="plotly_dark", height=420, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Feature Engineering Categories")
    st.markdown(
        """
        - **Trend:** SMA (10/20/50), EMA (10/20/50)
        - **Momentum / oscillators:** RSI (14), MACD + signal + histogram, momentum & ROC (10/20)
        - **Volatility:** Bollinger Bands (mid/upper/lower/%B), ATR (14), rolling volatility (10/20, annualized)
        - **Volume:** On-Balance Volume (OBV)
        - **Returns:** simple % return, log return
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- DATASET EXPLORER ----------------
with tab_explorer:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Raw Training Dataset")
    if train_df is None:
        st.warning(f"`{TRAIN_CSV}` not found.")
    else:
        st.caption(f"{train_df.shape[0]} rows × {train_df.shape[1]} columns")
        st.dataframe(train_df.tail(50), use_container_width=True, height=320)

        with st.expander("📊 Descriptive statistics (`.describe()`)"):
            st.dataframe(train_df.describe().T, use_container_width=True)

        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        default_cols = [c for c in ["adjusted", "close", "SMA_20", "EMA_20", "volume",
                                     "RSI_14", "MACD", "volatility_20", "wti_value", "gold_price"]
                         if c in numeric_cols]
        st.markdown("#### Correlation Heatmap")
        chosen_cols = st.multiselect("Columns to include", numeric_cols, default=default_cols)
        if len(chosen_cols) >= 2:
            corr = train_df[chosen_cols].corr()
            heat_fig = px.imshow(
                corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                aspect="auto",
            )
            heat_fig.update_layout(template="plotly_dark", height=520)
            st.plotly_chart(heat_fig, use_container_width=True)
        else:
            st.info("Pick at least 2 columns to render the heatmap.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Most Recent Daily Rows (`daily_latest.csv`)")
    if daily_df is not None:
        st.dataframe(daily_df, use_container_width=True, height=280)
    else:
        st.warning(f"`{DAILY_CSV}` not found.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    '<p class="footnote">Prototype dashboard — model metrics sourced from model1_proto.ipynb run logs; '
    'this reflects an experimental single-equity model, not a validated production system.</p>',
    unsafe_allow_html=True,
)
