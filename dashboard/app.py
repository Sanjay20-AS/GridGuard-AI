from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score
from huggingface_hub import hf_hub_download

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
DASHBOARD_DIR = BASE_DIR / "dashboard"

FEATURES_PATH = PROCESSED_DIR / "feature_test.npy"
LABELS_PATH = PROCESSED_DIR / "y_test.npy"
ATTACK_TYPES_PATH = PROCESSED_DIR / "attack_types.npy"
TEST_ERRORS_PATH = PROCESSED_DIR / "test_errors.npy"
TRAIN_ERRORS_PATH = PROCESSED_DIR / "train_errors.npy"
SHAP_VALUES_PATH = PROCESSED_DIR / "shap_values.npy"
FEATURE_NAMES_PATH = PROCESSED_DIR / "feature_test.csv"
AUTOENCODER_PATH = MODEL_DIR / "autoencoder.pt"
SURROGATE_PATH = MODEL_DIR / "surrogate_clf.pkl"
SCALER_PATHS = [MODEL_DIR / "scaler.pkl", MODEL_DIR / "minmax_scaler.joblib"]
SHAP_SUMMARY_PATH = DASHBOARD_DIR / "shap_summary.png"
SHAP_BAR_PATH = DASHBOARD_DIR / "shap_bar.png"

THRESHOLD_PERCENTILE = 25
RANDOM_STATE = 42

REPO_ID = "Sanjay-20/gridguard-ai"

def download_assets():
    files = [
        "models/autoencoder.pt",
        "models/surrogate_clf.pkl",
        "models/minmax_scaler.joblib",
        "data/processed/test_errors.npy",
        "data/processed/train_errors.npy",
        "data/processed/y_test.npy",
        "data/processed/attack_types.npy",
        "data/processed/feature_test.csv",
        "data/processed/shap_values.npy",
        "dashboard/shap_summary.png",
        "dashboard/shap_bar.png",
    ]
    for f in files:
        dest = BASE_DIR / f
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            hf_hub_download(
                repo_id=REPO_ID,
                filename=f,
                repo_type="model",
                local_dir=str(BASE_DIR)
            )
            print(f"Downloaded: {f}")

download_assets()


THRESHOLD_PERCENTILE = 25
RANDOM_STATE = 42

FEATURE_NAMES = [
    "mean_consumption",
    "median_consumption",
    "std_consumption",
    "min_consumption",
    "max_consumption",
    "p25_consumption",
    "p75_consumption",
    "skewness_consumption",
    "kurtosis_consumption",
    "peak_to_mean_ratio",
    "coefficient_of_variation",
    "quarter_1_mean",
    "quarter_2_mean",
    "quarter_3_mean",
    "quarter_4_mean",
    "trend_slope",
    "near_zero_days",
    "longest_near_zero_streak",
    "sudden_drop_count",
    "neighborhood_mean_diff",
    "neighborhood_std_diff",
    "neighborhood_mean_anomaly_score",
]

ATTACK_NAMES = {
    0: "Normal",
    1: "Scale Down",
    2: "Flatline",
    3: "Average",
    4: "Random Noise",
}

TOP_ATTACK_FEATURES = {
    1: ["sudden_drop_count", "quarter_1_mean", "min_consumption"],
    2: ["sudden_drop_count", "trend_slope", "quarter_1_mean"],
    3: ["skewness_consumption", "kurtosis_consumption", "peak_to_mean_ratio"],
    4: ["sudden_drop_count", "trend_slope", "quarter_1_mean"],
}


st.set_page_config(
    page_title="GridGuard AI — Energy Theft Detection Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .metric-card {
            background: linear-gradient(135deg, rgba(11, 29, 64, 0.96), rgba(18, 59, 92, 0.94));
            border-radius: 18px;
            padding: 18px 18px 14px 18px;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.18);
            border: 1px solid rgba(255, 255, 255, 0.08);
        }
        div[data-testid="stMetric"] {
            background: linear-gradient(135deg, rgba(11, 29, 64, 0.96), rgba(18, 59, 92, 0.94));
            border-radius: 18px;
            padding: 14px 16px;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.18);
            border: 1px solid rgba(255, 255, 255, 0.08);
        }
        div[data-testid="stMetricLabel"], div[data-testid="stMetricValue"], div[data-testid="stMetricDelta"] {
            color: #ffffff;
        }
        .metric-card .label {
            color: #9cc9d9;
            font-size: 0.92rem;
            margin-bottom: 6px;
        }
        .metric-card .value {
            color: #ffffff;
            font-size: 2rem;
            font-weight: 700;
            line-height: 1.1;
        }
        .metric-card .delta {
            color: #8fe3c8;
            font-size: 0.84rem;
            margin-top: 6px;
        }
        .block-container {
            padding-top: 1.2rem;
        }
        .stRadio > div {
            gap: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_numpy_arrays() -> Dict[str, np.ndarray]:
    """Load all dashboard data once and reuse it across interactions."""
    return {
        "features": np.load(FEATURES_PATH),
        "labels": np.load(LABELS_PATH),
        "attack_types": np.load(ATTACK_TYPES_PATH),
        "test_errors": np.load(TEST_ERRORS_PATH),
        "shap_values": np.load(SHAP_VALUES_PATH),
    }


@st.cache_data(show_spinner=False)
def load_feature_names() -> List[str]:
    """Read the feature columns from the CSV header."""
    return list(pd.read_csv(FEATURE_NAMES_PATH, nrows=0).columns)


@st.cache_resource(show_spinner=False)
def load_surrogate_model():
    """Load the surrogate classifier used for live predictions."""
    return joblib.load(SURROGATE_PATH)


@st.cache_resource(show_spinner=False)
def load_scaler():
    """Load the fitted MinMaxScaler used in preprocessing."""
    for scaler_path in SCALER_PATHS:
        if scaler_path.exists():
            return joblib.load(scaler_path)
    return None


@st.cache_data(show_spinner=False)
def load_train_errors() -> np.ndarray:
    """Load train reconstruction errors for threshold selection."""
    return np.load(TRAIN_ERRORS_PATH)


@st.cache_data(show_spinner=False)
def load_threshold_metrics() -> Dict[str, float]:
    """Compute the 25th-percentile threshold and the resulting test metrics."""
    data = load_numpy_arrays()
    train_errors = load_train_errors()
    threshold = float(np.percentile(train_errors, THRESHOLD_PERCENTILE))
    predictions = (data["test_errors"] > threshold).astype(int)

    precision = float(precision_score(data["labels"], predictions, zero_division=0) * 100.0)
    recall = float(recall_score(data["labels"], predictions, zero_division=0) * 100.0)
    f1 = float(f1_score(data["labels"], predictions, zero_division=0) * 100.0)
    flagged = int(np.sum(predictions))

    return {
        "threshold": threshold,
        "flagged": flagged,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


@st.cache_data(show_spinner=False)
def load_images() -> Dict[str, Image.Image]:
    """Load SHAP plot images from disk."""
    return {
        "summary": Image.open(SHAP_SUMMARY_PATH),
        "bar": Image.open(SHAP_BAR_PATH),
    }


@st.cache_data(show_spinner=False)
def get_customer_shap_values(customer_index: int) -> np.ndarray:
    """Load SHAP values for a specific customer, computing on-the-fly if needed."""
    data = load_numpy_arrays()
    shap_values = data["shap_values"]

    if shap_values.shape[0] == data["features"].shape[0]:
        return shap_values[customer_index]

    model = load_surrogate_model()
    explainer = shap.TreeExplainer(model)
    row_features = data["features"][customer_index:customer_index+1]
    raw_values = explainer.shap_values(row_features)
    if isinstance(raw_values, list):
        customer_shap = np.asarray(raw_values[1] if len(raw_values) > 1 else raw_values[0])[0]
    else:
        customer_shap = np.asarray(raw_values)
        if customer_shap.ndim == 3:
            customer_shap = customer_shap[0, :, 1]
        else:
            customer_shap = customer_shap[0]
            
    return customer_shap


@st.cache_data(show_spinner=False)
def build_attack_distribution() -> pd.DataFrame:
    """Build the attack-type counts used in the overview tab."""
    data = load_numpy_arrays()
    attack_types = data["attack_types"]
    counts = []
    for attack_id, attack_name in ATTACK_NAMES.items():
        counts.append({"attack_id": attack_id, "attack_name": attack_name, "count": int(np.sum(attack_types == attack_id))})
    return pd.DataFrame(counts)


@st.cache_data(show_spinner=False)
def build_customer_table(customer_index: int) -> pd.DataFrame:
    """Return a single customer's feature vector as a table."""
    data = load_numpy_arrays()
    feature_names = load_feature_names()
    row = data["features"][customer_index]
    return pd.DataFrame({"Feature": feature_names, "Value": row})


@st.cache_data(show_spinner=False)
def get_customer_prediction(customer_index: int, threshold: float | None = None) -> Dict[str, object]:
    """Return the customer error and classification verdict."""
    data = load_numpy_arrays()
    threshold_value = load_threshold_metrics()["threshold"] if threshold is None else threshold
    error = float(data["test_errors"][customer_index])
    prediction = "theft" if error > threshold_value else "normal"
    return {
        "error": error,
        "prediction": prediction,
        "attack_type": int(data["attack_types"][customer_index]),
        "actual_label": int(data["labels"][customer_index]),
    }


@st.cache_data(show_spinner=False)
def build_top_attack_table() -> pd.DataFrame:
    """Hardcode the top-three features per attack type from the explainability results."""
    rows = []
    for attack_id, features in TOP_ATTACK_FEATURES.items():
        rows.append(
            {
                "Attack Type": f"{attack_id} - {ATTACK_NAMES[attack_id]}",
                "Top Feature 1": features[0],
                "Top Feature 2": features[1],
                "Top Feature 3": features[2],
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def get_feature_index_map() -> Dict[str, int]:
    """Map feature names to column indices for live simulation."""
    return {name: index for index, name in enumerate(load_feature_names())}


def render_metric_card(label: str, value: str, delta: str = "") -> None:
    """Render a stylized metric card."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="delta">{delta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_donut_chart(labels: np.ndarray) -> go.Figure:
    """Create a donut chart for normal vs theft distribution."""
    normal_count = int(np.sum(labels == 0))
    theft_count = int(np.sum(labels == 1))
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Normal", "Theft"],
                values=[normal_count, theft_count],
                hole=0.55,
                marker=dict(colors=["#1f9d7a", "#0b1d40"]),
                textinfo="label+percent",
            )
        ]
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def build_attack_bar_chart() -> go.Figure:
    """Create a bar chart of attack type counts."""
    df = build_attack_distribution()
    fig = go.Figure(
        data=[
            go.Bar(
                x=df["attack_name"],
                y=df["count"],
                marker_color=["#1f9d7a", "#0d5b7a", "#1b7f6a", "#3aa6b9"],
                text=df["count"],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="Customers by Attack Type",
        xaxis_title="Attack Type",
        yaxis_title="Count",
        margin=dict(l=10, r=10, t=50, b=10),
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_customer_shap_chart(customer_index: int) -> go.Figure:
    """Create a bar chart of the customer's top SHAP contributions."""
    data = load_numpy_arrays()
    feature_names = load_feature_names()
    row = get_customer_shap_values(customer_index)
    top_indices = np.argsort(np.abs(row))[::-1][:5]
    top_features = [feature_names[idx] for idx in top_indices]
    top_values = [float(row[idx]) for idx in top_indices]
    colors = ["#0b1d40" if value < 0 else "#1f9d7a" for value in top_values]

    fig = go.Figure(
        data=[
            go.Bar(
                x=top_features,
                y=top_values,
                marker_color=colors,
            )
        ]
    )
    fig.update_layout(
        title="Top 5 SHAP Contributions",
        xaxis_title="Feature",
        yaxis_title="Contribution",
        margin=dict(l=10, r=10, t=50, b=10),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def get_live_prediction(feature_updates: Dict[str, float]) -> tuple[float, str]:
    """Predict theft probability from the surrogate using a modified feature row."""
    model = load_surrogate_model()
    scaler = load_scaler()
    data = load_numpy_arrays()
    feature_names = load_feature_names()
    feature_index_map = get_feature_index_map()

    base_row = data["features"][0].copy()
    for feature_name, value in feature_updates.items():
        if feature_name in feature_index_map:
            base_row[feature_index_map[feature_name]] = value

    # The surrogate works on engineered feature values; the scaler is loaded for completeness
    # and future reuse in the dashboard, but the live simulation predicts directly on the feature vector.
    _ = scaler

    probability = float(model.predict_proba(base_row.reshape(1, -1))[0][1])
    verdict = "theft" if probability >= 0.5 else "normal"
    return probability, verdict


st.sidebar.title("GridGuard AI")
st.sidebar.markdown(
    "Energy theft detection with an LSTM autoencoder, a surrogate classifier, and SHAP-based explanations."
)
selected_tab = st.sidebar.radio(
    "Navigate",
    ["Overview", "Model Insights", "Customer Lookup", "Live Simulation"],
)

with st.sidebar.expander("About this project"):
    st.write(
        "The autoencoder learns normal electricity usage patterns and flags unusual reconstruction behavior. "
        "A surrogate classifier and SHAP are used to explain the model in simple feature-level terms."
    )

st.title("GridGuard AI — Energy Theft Detection Dashboard")
st.caption("Monitoring consumption patterns, reconstruction anomalies, and feature-level explanations.")

data = load_numpy_arrays()
labels = data["labels"]
attack_types = data["attack_types"]
threshold_metrics = load_threshold_metrics()

if selected_tab == "Overview":
    total_customers = len(labels)
    flagged_customers = int(threshold_metrics["flagged"])
    st.markdown("### Overview")
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("Total Customers Monitored", f"{total_customers:,}", "Processed test customers")
    with metric_cols[1]:
        st.metric("Flagged as Theft", f"{flagged_customers:,}", f"Threshold: P{THRESHOLD_PERCENTILE} = {threshold_metrics['threshold']:.6f}")
    with metric_cols[2]:
        st.metric("Precision", f"{threshold_metrics['precision']:.1f}%", f"F1: {threshold_metrics['f1']:.1f}%")
    with metric_cols[3]:
        st.metric("Recall", f"{threshold_metrics['recall']:.1f}%", f"P{THRESHOLD_PERCENTILE} threshold")

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.plotly_chart(build_donut_chart(labels), width="stretch")
    with chart_cols[1]:
        st.plotly_chart(build_attack_bar_chart(), width="stretch")

elif selected_tab == "Model Insights":
    st.markdown("### Model Insights")
    image_cols = st.columns(2)
    with image_cols[0]:
        st.image(str(SHAP_SUMMARY_PATH), caption="SHAP summary plot", width="stretch")
    with image_cols[1]:
        st.image(str(SHAP_BAR_PATH), caption="SHAP bar plot", width="stretch")

    st.markdown(
        """
        **What the SHAP plots mean**

        These charts show which features the model leans on most when deciding whether usage looks normal or suspicious. Larger bars mean a feature had a stronger influence on the prediction, while the summary plot shows how each feature behaves across many customers.

        In simple terms, the model focuses most on the strongest consumption pattern differences, especially features that separate steady households from unusual tampering behavior.
        """
    )

    st.markdown("#### Top 3 features per attack type")
    st.table(build_top_attack_table())

elif selected_tab == "Customer Lookup":
    st.markdown("### Customer Lookup")
    max_index = len(labels) - 1
    customer_index = st.number_input("Choose a customer index", min_value=0, max_value=max_index, value=0, step=1)
    customer_index = int(customer_index)

    verdict = get_customer_prediction(customer_index)
    verdict_color = "#1f9d7a" if verdict["prediction"] == "normal" else "#c0392b"

    st.markdown(
        f"""
        <div style="padding:12px 16px;border-radius:14px;background:{verdict_color};color:white;font-weight:700;display:inline-block;">
            Verdict: {verdict['prediction'].upper()}
        </div>
        """,
        unsafe_allow_html=True,
    )

    display_cols = st.columns(3)
    with display_cols[0]:
        st.metric("Reconstruction Error", f"{verdict['error']:.6f}")
    with display_cols[1]:
        st.metric("Actual Label", "Theft" if verdict["actual_label"] == 1 else "Normal")
    with display_cols[2]:
        st.metric("Attack Type", ATTACK_NAMES.get(verdict["attack_type"], "Normal"))

    st.markdown("#### Feature values")
    st.dataframe(build_customer_table(customer_index), width="stretch", height=420)

    st.plotly_chart(build_customer_shap_chart(customer_index), width="stretch")

elif selected_tab == "Live Simulation":
    st.markdown("### Live Simulation")
    st.write("Adjust a few key indicators to see how the surrogate classifier responds in real time.")

    feature_names = load_feature_names()
    feature_index_map = get_feature_index_map()
    base_features = data["features"][0].copy()

    control_cols = st.columns(2)
    with control_cols[0]:
        val_cv = round(float(base_features[feature_index_map["coefficient_of_variation"]]), 2)
        coeff_var = st.slider("coefficient_of_variation", 0.0, 3.0, val_cv, 0.01)
        
        val_min = round(float(base_features[feature_index_map["min_consumption"]]), 2)
        min_consumption = st.slider("min_consumption", 0.0, 1.0, val_min, 0.01)
        
    with control_cols[1]:
        val_drops = round(float(base_features[feature_index_map["sudden_drop_count"]]), 0)
        sudden_drops = st.slider("sudden_drop_count", 0.0, 80.0, val_drops, 1.0)
        
        val_peak = round(float(base_features[feature_index_map["peak_to_mean_ratio"]]), 2)
        peak_to_mean = st.slider("peak_to_mean_ratio", 0.0, 10.0, val_peak, 0.01)

    feature_updates = {
        "coefficient_of_variation": coeff_var,
        "min_consumption": min_consumption,
        "sudden_drop_count": sudden_drops,
        "peak_to_mean_ratio": peak_to_mean,
    }
    probability, verdict = get_live_prediction(feature_updates)

    st.metric("Theft Probability", f"{probability * 100:.1f}%")
    st.progress(min(max(probability, 0.0), 1.0))

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#0b1d40" if verdict == "theft" else "#1f9d7a"},
                "steps": [
                    {"range": [0, 50], "color": "#eaf4f4"},
                    {"range": [50, 100], "color": "#dceef5"},
                ],
            },
            title={"text": "Theft Risk"},
        )
    )
    gauge.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(gauge, width="stretch")

    st.info(f"Predicted verdict: {verdict.upper()}")

else:
    st.error("Unknown tab selection.")
