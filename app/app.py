import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ---------------------------------------------------------------------------
# Project Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_COMPARISON_PATH = DATA_DIR / "model_comparison.csv"

REGRESSION_MODELS = [
    "XGBoost",
    "Random Forest",
    "SVR",
    "KNN",
    "Linear Regression"
]

CLUSTER_LABELS = {
    0: "Low Performance",
    1: "High Performance",
    2: "Moderate Performance"
}


# ---------------------------------------------------------------------------
# Cached Model + Data Loading (unchanged logic)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_models():

    model_files = {
        "Linear Regression": "linear_regression.joblib",
        "KNN": "knn.joblib",
        "SVR": "svr.joblib",
        "Random Forest": "random_forest.joblib",
        "XGBoost": "xgboost.joblib",
        "K-Means": "kmeans.joblib"
    }

    return {
        model_name: joblib.load(MODELS_DIR / file_name)
        for model_name, file_name in model_files.items()
    }


@st.cache_data
def load_model_comparison():
    return pd.read_csv(MODEL_COMPARISON_PATH)


model_comparison = load_model_comparison()
models = load_models()


@st.cache_data
def get_dataset_record_count():
    """Total dataset size = train rows + test rows, when available.

    Falls back to the last known record count if the train/test files
    are missing, so a change in the data directory can never break
    application startup.
    """
    try:
        train_rows = len(pd.read_csv(DATA_DIR / "X_train.csv"))
        test_rows = len(pd.read_csv(DATA_DIR / "X_test.csv"))
        return train_rows + test_rows
    except Exception:
        return 16829


DATASET_RECORD_COUNT = get_dataset_record_count()


# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="NetPredict",
    page_icon="•",
    layout="wide"
)


# ---------------------------------------------------------------------------
# Design System (theme-adaptive CSS)
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600;700&display=swap');

    :root {
        --np-accent: #2F6FED;
        --np-accent-soft: rgba(47, 111, 237, 0.10);
        --np-accent-border: rgba(47, 111, 237, 0.35);
        --np-success: #12946B;
        --np-success-soft: rgba(18, 148, 107, 0.12);
        --np-success-border: rgba(18, 148, 107, 0.40);
        --np-warning: #B7791F;
        --np-warning-soft: rgba(183, 121, 31, 0.12);
        --np-warning-border: rgba(183, 121, 31, 0.40);
        --np-danger: #C0392B;
        --np-danger-soft: rgba(192, 57, 43, 0.12);
        --np-danger-border: rgba(192, 57, 43, 0.40);
        --np-border: rgba(128, 128, 128, 0.22);
        --np-surface: rgba(128, 128, 128, 0.05);
        --np-radius: 10px;
    }

    html, body, [class*="css"] {
        font-family: "IBM Plex Sans", "Source Sans Pro", sans-serif;
    }

    h1, h2, h3, h4 {
        font-family: "IBM Plex Sans", sans-serif;
        letter-spacing: -0.01em;
    }

    [data-testid="stMetricValue"] {
        font-family: "IBM Plex Mono", monospace;
    }

    hr { margin: 1.15rem 0; }

    /* ---- Sidebar ---- */

    [data-testid="stSidebar"] {
        border-right: 1px solid var(--np-border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }

    .np-brand {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        padding: 0 0.2rem 1rem 0.2rem;
    }

    .np-brand-mark {
        width: 34px;
        height: 34px;
        border-radius: 8px;
        background: var(--np-accent-soft);
        border: 1px solid var(--np-accent-border);
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        font-family: "IBM Plex Mono", monospace;
        font-weight: 700;
        font-size: 0.85rem;
        color: var(--np-accent);
    }

    .np-brand-title {
        font-size: 1.1rem;
        font-weight: 700;
        line-height: 1.15;
        margin: 0;
    }

    .np-brand-subtitle {
        font-size: 0.74rem;
        opacity: 0.6;
        margin-top: 0.1rem;
    }

    .np-nav-active {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        padding: 0.55rem 0.75rem;
        margin-bottom: 0.3rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
        background: var(--np-accent-soft);
        border-left: 3px solid var(--np-accent);
    }

    .np-nav-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--np-accent);
        flex-shrink: 0;
    }

    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        text-align: left;
        justify-content: flex-start;
        background: transparent;
        border: none;
        border-left: 3px solid transparent;
        border-radius: 8px;
        padding: 0.55rem 0.75rem;
        font-size: 0.9rem;
        font-weight: 500;
        opacity: 0.82;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background: var(--np-surface);
        opacity: 1;
        border-left-color: var(--np-border);
        color: inherit;
    }

    .np-sidebar-footer {
        font-size: 0.72rem;
        opacity: 0.55;
        line-height: 1.6;
        margin-top: 0.25rem;
    }

    /* ---- Hero ---- */

    .np-hero-title {
        font-size: 2.15rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }

    .np-hero-subtitle {
        font-size: 1.05rem;
        font-weight: 500;
        opacity: 0.72;
        margin-top: 0.3rem;
    }

    .np-hero-desc {
        font-size: 0.95rem;
        opacity: 0.72;
        max-width: 72ch;
        margin-top: 0.7rem;
        line-height: 1.6;
    }

    /* ---- Section headers ---- */

    .np-section-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin: 0.1rem 0 0.15rem 0;
    }

    .np-section-subtitle {
        font-size: 0.87rem;
        opacity: 0.62;
        margin-bottom: 0.7rem;
        line-height: 1.5;
    }

    /* ---- Cards ---- */

    .np-card-title {
        font-weight: 600;
        font-size: 0.92rem;
        margin-bottom: 0.35rem;
    }

    .np-insight-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 0.1rem;
    }

    .np-insight-metric {
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--np-accent);
        margin-bottom: 0.55rem;
    }

    .np-card-body {
        font-size: 0.85rem;
        opacity: 0.72;
        line-height: 1.55;
    }

    .np-step-row {
        display: flex;
        align-items: flex-start;
        gap: 0.85rem;
    }

    .np-step-index {
        font-family: "IBM Plex Mono", monospace;
        font-weight: 600;
        font-size: 0.78rem;
        color: var(--np-accent);
        background: var(--np-accent-soft);
        border: 1px solid var(--np-accent-border);
        width: 26px;
        height: 26px;
        min-width: 26px;
        border-radius: 7px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-top: 0.05rem;
    }

    /* ---- Badges ---- */

    .np-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.7rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid transparent;
    }

    .np-badge-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
    }

    /* ---- Tech pills ---- */

    .np-pill-row { display: flex; flex-wrap: wrap; gap: 0.5rem; }

    .np-pill {
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.8rem;
        padding: 0.35rem 0.75rem;
        border-radius: 7px;
        border: 1px solid var(--np-border);
        background: var(--np-surface);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------------------------
# Shared UI helpers
# ---------------------------------------------------------------------------

def section_header(title, subtitle=None):
    st.markdown(f'<div class="np-section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="np-section-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def render_card(title, body):
    with st.container(border=True):
        st.markdown(f'<div class="np-card-title">{title}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="np-card-body">{body}</div>', unsafe_allow_html=True)


def render_step(index, title, body):
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="np-step-row">
                <div class="np-step-index">{index:02d}</div>
                <div>
                    <div class="np-card-title">{title}</div>
                    <div class="np-card-body">{body}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


def performance_tone(level_text):
    if "Low" in level_text:
        return "low"
    if "Moderate" in level_text:
        return "moderate"
    if "High" in level_text:
        return "high"
    return "neutral"


def render_badge(label, tone="neutral"):
    tones = {
        "low": ("var(--np-danger)", "var(--np-danger-soft)", "var(--np-danger-border)"),
        "moderate": ("var(--np-warning)", "var(--np-warning-soft)", "var(--np-warning-border)"),
        "high": ("var(--np-success)", "var(--np-success-soft)", "var(--np-success-border)"),
        "neutral": ("var(--np-accent)", "var(--np-accent-soft)", "var(--np-accent-border)"),
    }
    color, soft, border = tones.get(tone, tones["neutral"])
    st.markdown(
        f"""
        <span class="np-badge" style="color:{color};background:{soft};border-color:{border};">
            <span class="np-badge-dot" style="background:{color};"></span>{label}
        </span>
        """,
        unsafe_allow_html=True
    )


def styled_metric_table(df, highlight_col="MAE", ascending=True):
    """Return a pandas Styler with consistent decimal formatting and the
    best row (by highlight_col) highlighted."""
    best_index = df[highlight_col].idxmin() if ascending else df[highlight_col].idxmax()

    def highlight_row(row):
        if row.name == best_index:
            return ["background-color: var(--np-success-soft)"] * len(row)
        return [""] * len(row)

    fmt = {}
    for col in ("MAE", "RMSE", "R²"):
        if col in df.columns:
            fmt[col] = "{:.4f}"

    return df.style.apply(highlight_row, axis=1).format(fmt)


# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------

st.sidebar.markdown(
    """
    <div class="np-brand">
        <div class="np-brand-mark">NP</div>
        <div>
            <p class="np-brand-title">NetPredict</p>
            <p class="np-brand-subtitle">Network Throughput Analytics</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.divider()

navigation_items = [
    "Overview",
    "Prediction",
    "Model Comparison",
    "Network Segmentation",
    "Model Evaluation",
    "About"
]

if "page" not in st.session_state:
    st.session_state.page = "Overview"

for page_name in navigation_items:

    is_active = st.session_state.page == page_name

    if is_active:
        st.sidebar.markdown(
            f"""
            <div class="np-nav-active">
                <span class="np-nav-dot"></span>{page_name}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        if st.sidebar.button(page_name, key=f"nav_{page_name}", use_container_width=True):
            st.session_state.page = page_name
            st.rerun()

page = st.session_state.page

st.sidebar.divider()

st.sidebar.markdown(
    """
    <div class="np-sidebar-footer">
        Machine Learning Project<br>
        Regression • Clustering • Evaluation
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------------------------
# Overview Page
# ---------------------------------------------------------------------------

if page == "Overview":

    st.markdown(
        """
        <div class="np-hero-title">NetPredict</div>
        <div class="np-hero-subtitle">Network Throughput Prediction &amp; Performance Analytics</div>
        <div class="np-hero-desc">
            NetPredict is a machine learning platform that estimates data throughput
            from real-world network conditions and geographic context. It combines
            five regression models with K-Means clustering to both predict expected
            throughput and classify the resulting network performance level.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    section_header("Project at a Glance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        with st.container(border=True):
            st.metric("Dataset Records", f"{DATASET_RECORD_COUNT:,}")

    with col2:
        with st.container(border=True):
            st.metric("Regression Models", "5")

    with col3:
        with st.container(border=True):
            st.metric("Performance Clusters", "3")

    with col4:
        best_overall = model_comparison.sort_values("MAE", ascending=True).iloc[0]
        with st.container(border=True):
            st.metric("Selected Model", best_overall["Model"])

    st.divider()

    section_header(
        "Model Performance Snapshot",
        "Regression models evaluated on the same held-out test set, ranked by MAE."
    )

    overview_comparison = model_comparison.sort_values(by="MAE", ascending=True).reset_index(drop=True)

    st.dataframe(
        styled_metric_table(overview_comparison, highlight_col="MAE", ascending=True),
        use_container_width=True,
        hide_index=True
    )

    best_mae_row = model_comparison.sort_values("MAE").iloc[0]
    best_rmse_row = model_comparison.sort_values("RMSE").iloc[0]
    best_r2_row = model_comparison.sort_values("R²", ascending=False).iloc[0]

    if best_mae_row["Model"] == best_rmse_row["Model"] == best_r2_row["Model"]:
        snapshot_note = (
            f"{best_mae_row['Model']} leads on every metric and is used as the "
            "default model for prediction."
        )
    else:
        snapshot_note = (
            f"{best_mae_row['Model']} has the lowest MAE and is selected as the "
            f"primary model, while {best_rmse_row['Model']} shows a marginally "
            f"lower RMSE and {best_r2_row['Model']} a marginally higher R² — the "
            "models perform closely overall."
        )

    st.caption(snapshot_note)

    st.divider()

    section_header("Primary Model Insight")

    with st.container(border=True):
        st.markdown(f'<div class="np-insight-title">{best_mae_row["Model"]}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="np-insight-metric">Lowest Test MAE: {best_mae_row["MAE"]:.4f} Mbps</div>',
            unsafe_allow_html=True
        )
        if best_mae_row["Model"] == best_rmse_row["Model"] == best_r2_row["Model"]:
            insight_text = (
                f"{best_mae_row['Model']} was selected as the primary regression "
                "model because it achieved the best result across MAE, RMSE, and R²."
            )
        else:
            insight_text = (
                f"{best_mae_row['Model']} was selected as the primary regression "
                "model because it achieved the lowest test MAE among the evaluated "
                f"models. {best_rmse_row['Model']} achieved a slightly lower RMSE and "
                f"{best_r2_row['Model']} a slightly higher R², indicating that model "
                "performance is close across the evaluated approaches."
            )
        st.markdown(f'<div class="np-card-body">{insight_text}</div>', unsafe_allow_html=True)

    st.divider()

    section_header("Project Objectives")

    obj_col1, obj_col2 = st.columns(2)

    with obj_col1:
        render_card(
            "Throughput Prediction",
            "Predict network throughput in Mbps using multiple regression "
            "models trained on network condition and location data."
        )
        render_card(
            "Model Comparison",
            "Compare regression models side by side using MAE, RMSE, and R² "
            "to understand their relative strengths."
        )

    with obj_col2:
        render_card(
            "Network Segmentation",
            "Group network observations into performance clusters using "
            "K-Means, based on signal strength, throughput, and latency."
        )
        render_card(
            "Performance Analysis",
            "Examine how network conditions relate to observed throughput "
            "and where predictions may fall short."
        )


# ---------------------------------------------------------------------------
# Prediction Page
# ---------------------------------------------------------------------------

if page == "Prediction":

    st.title("Network Throughput Prediction")
    st.caption("Enter the current network conditions to estimate expected data throughput.")

    st.divider()

    model_name = st.selectbox("Regression Model", REGRESSION_MODELS)

    section_header("Location")

    loc_col1, loc_col2, loc_col3 = st.columns(3)

    with loc_col1:
        locality = st.selectbox(
            "Locality",
            [
                "Anandpuri", "Anisabad", "Ashok Rajpath", "Bailey Road",
                "Bankipore", "Boring Canal Road", "Boring Road", "Danapur",
                "Exhibition Road", "Fraser Road", "Gandhi Maidan", "Gardanibagh",
                "Kankarbagh", "Kidwaipuri", "Kumrhar", "Pataliputra",
                "Patliputra Colony", "Phulwari Sharif", "Rajendra Nagar", "S.K. Puri"
            ]
        )

    with loc_col2:
        latitude = st.number_input("Latitude", value=25.5900, format="%.4f")

    with loc_col3:
        longitude = st.number_input("Longitude", value=85.1300, format="%.4f")

    section_header("Network Conditions")

    net_col1, net_col2, net_col3 = st.columns(3)

    with net_col1:
        network_type = st.selectbox("Network Type", ["3G", "4G", "5G"])

    with net_col2:
        signal_strength = st.number_input(
            "Signal Strength (dBm)",
            min_value=-150.0, max_value=0.0, value=-90.0, format="%.2f"
        )

    with net_col3:
        latency = st.number_input(
            "Latency (ms)",
            min_value=0.0, max_value=1000.0, value=100.0, format="%.2f"
        )

    st.divider()

    predict_button = st.button("Predict Throughput", use_container_width=True, type="primary")

    if not predict_button:
        st.caption("Results will appear here once a prediction is run.")

    if predict_button:

        input_data = pd.DataFrame({
            "Locality": [locality],
            "Latitude": [latitude],
            "Longitude": [longitude],
            "Signal Strength (dBm)": [signal_strength],
            "Latency (ms)": [latency],
            "Network Type": [network_type]
        })

        selected_model = models[model_name]
        prediction = selected_model.predict(input_data)
        prediction = float(np.asarray(prediction).ravel()[0])

        kmeans_input = pd.DataFrame({
            "Signal Strength (dBm)": [signal_strength],
            "Data Throughput (Mbps)": [prediction],
            "Latency (ms)": [latency]
        })

        kmeans_model = models["K-Means"]
        cluster = int(kmeans_model.predict(kmeans_input)[0])
        performance_level = CLUSTER_LABELS[cluster]

        model_mae = model_comparison.loc[model_comparison["Model"] == model_name, "MAE"].iloc[0]

        st.divider()

        section_header("Prediction Result")

        res_col1, res_col2, res_col3, res_col4 = st.columns(4)

        with res_col1:
            with st.container(border=True):
                st.metric("Selected Model", model_name)

        with res_col2:
            with st.container(border=True):
                st.metric("Predicted Throughput", f"{prediction:.2f} Mbps")

        with res_col3:
            with st.container(border=True):
                st.markdown('<div class="np-card-title">Performance Level</div>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                render_badge(performance_level, performance_tone(performance_level))

        with res_col4:
            with st.container(border=True):
                st.metric("Typical Absolute Error", f"~{model_mae:.4f} Mbps")

        st.caption(
            "MAE summarizes average absolute prediction error on the test set "
            "and does not represent a guaranteed error range for an individual "
            "prediction."
        )

        st.divider()

        section_header("Input Summary")

        st.dataframe(input_data, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Model Comparison Page
# ---------------------------------------------------------------------------

if page == "Model Comparison":

    st.title("Model Comparison")
    st.caption("All regression models were evaluated on the same held-out test dataset.")

    st.divider()

    comparison_df = model_comparison.copy()

    best_mae_row = comparison_df.sort_values("MAE").iloc[0]
    best_rmse_row = comparison_df.sort_values("RMSE").iloc[0]
    best_r2_row = comparison_df.sort_values("R²", ascending=False).iloc[0]

    section_header("Key Results")

    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        with st.container(border=True):
            st.metric("Best Model (MAE)", best_mae_row["Model"])

    with kpi_col2:
        with st.container(border=True):
            st.metric("Lowest MAE", f'{best_mae_row["MAE"]:.4f}')

    with kpi_col3:
        with st.container(border=True):
            st.metric("Lowest RMSE", f'{best_rmse_row["RMSE"]:.4f}', help=f'Achieved by {best_rmse_row["Model"]}')

    with kpi_col4:
        with st.container(border=True):
            st.metric("Highest R²", f'{best_r2_row["R²"]:.4f}', help=f'Achieved by {best_r2_row["Model"]}')

    st.divider()

    section_header("Ranking by MAE", "Lower MAE indicates better average prediction accuracy.")

    rank_df = comparison_df.sort_values("MAE", ascending=True).reset_index(drop=True)
    rank_df["Rank"] = range(1, len(rank_df) + 1)

    cols = ["Rank"] + [col for col in rank_df.columns if col != "Rank"]
    rank_df = rank_df[cols]

    st.dataframe(
        styled_metric_table(rank_df, highlight_col="MAE", ascending=True),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    section_header("Metric Comparison", "Each chart uses a zero-based scale to avoid a misleading visual impression.")

    chart_mae, chart_rmse, chart_r2 = st.tabs(["MAE (lower is better)", "RMSE (lower is better)", "R² (higher is better)"])

    with chart_mae:
        st.bar_chart(comparison_df.set_index("Model")[["MAE"]])

    with chart_rmse:
        st.bar_chart(comparison_df.set_index("Model")[["RMSE"]])

    with chart_r2:
        st.bar_chart(comparison_df.set_index("Model")[["R²"]])

    st.divider()

    section_header("Best Model Analysis")

    if best_mae_row["Model"] == best_rmse_row["Model"] == best_r2_row["Model"]:
        analysis_text = (
            f"**{best_mae_row['Model']}** achieved the best result across MAE, RMSE, "
            "and R², and was therefore selected as the primary regression model."
        )
    else:
        analysis_text = (
            f"**{best_mae_row['Model']}** achieved the lowest MAE "
            f"({best_mae_row['MAE']:.4f}) among the evaluated models and was "
            f"therefore selected as the primary regression model. **{best_rmse_row['Model']}** "
            f"recorded a slightly lower RMSE ({best_rmse_row['RMSE']:.4f}), and "
            f"**{best_r2_row['Model']}** produced a marginally higher R² "
            f"({best_r2_row['R²']:.4f}). The gap between models is small, and "
            "Linear Regression in particular remains competitive despite being "
            "the simplest model in the comparison."
        )

    st.markdown(analysis_text)

    st.divider()

    section_header("Performance Insights", "Best-per-metric results, calculated directly from the comparison data.")

    worst_mae = comparison_df["MAE"].max()
    best_mae = comparison_df["MAE"].min()
    mae_improvement_pct = (worst_mae - best_mae) / worst_mae * 100

    insight_col1, insight_col2, insight_col3, insight_col4 = st.columns(4)

    with insight_col1:
        with st.container(border=True):
            st.metric("Best MAE", f'{best_mae_row["MAE"]:.4f}', help=f'Achieved by {best_mae_row["Model"]}')

    with insight_col2:
        with st.container(border=True):
            st.metric("Best RMSE", f'{best_rmse_row["RMSE"]:.4f}', help=f'Achieved by {best_rmse_row["Model"]}')

    with insight_col3:
        with st.container(border=True):
            st.metric("Best R²", f'{best_r2_row["R²"]:.4f}', help=f'Achieved by {best_r2_row["Model"]}')

    with insight_col4:
        with st.container(border=True):


            st.metric("MAE Improvement", f"{mae_improvement_pct:.2f}%")

    st.caption(f"Best MAE is {mae_improvement_pct:.2f}% lower than the worst MAE among the evaluated models.")

    st.markdown(
        "Model performance is relatively close across the evaluated algorithms, "
        f"with {best_mae_row['Model']} providing the lowest MAE and therefore "
        "being selected as the primary model."
    )

    st.divider()

    section_header("Model Interpretation")

    desc_col1, desc_col2, desc_col3 = st.columns(3)

    with desc_col1:
        render_card("Linear Regression", "Baseline linear relationship between features and throughput.")
        render_card("Random Forest", "Ensemble of decision trees trained on bootstrapped samples.")

    with desc_col2:
        render_card("KNN", "Distance-based prediction using neighboring observations.")
        render_card("XGBoost", "Gradient boosting model using sequential tree ensembles.")

    with desc_col3:
        render_card("SVR", "Non-linear regression using the RBF kernel.")


# ---------------------------------------------------------------------------
# Network Segmentation Page
# ---------------------------------------------------------------------------

if page == "Network Segmentation":

    st.title("Network Performance Segmentation")
    st.caption(
        "K-Means clustering groups network observations into performance segments "
        "based on signal strength, throughput, and latency."
    )

    st.divider()

    section_header(
        "Performance Cluster Profiles",
        "Labels are interpretations of each cluster's average profile, not causal explanations."
    )

    cluster_profiles = pd.DataFrame({
        "Performance Level": ["Low Performance", "Moderate Performance", "High Performance"],
        "Signal Strength (dBm)": [-86.47, -91.27, -95.82],
        "Throughput (Mbps)": [2.12, 7.56, 62.98],
        "Latency (ms)": [153.21, 77.90, 29.61]
    })

    st.dataframe(
        cluster_profiles.style.format({
            "Signal Strength (dBm)": "{:.2f}",
            "Throughput (Mbps)": "{:.2f}",
            "Latency (ms)": "{:.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    section_header("Throughput by Performance Level")

    throughput_profile = cluster_profiles[["Performance Level", "Throughput (Mbps)"]].set_index("Performance Level")
    st.bar_chart(throughput_profile)

    section_header("Latency by Performance Level")

    latency_profile = cluster_profiles[["Performance Level", "Latency (ms)"]].set_index("Performance Level")
    st.bar_chart(latency_profile)
    st.caption(
        "Latency varies substantially across the observed cluster profiles, "
        "helping distinguish the performance segments."
    )

    st.divider()

    section_header("Identify a Performance Cluster")

    seg_col1, seg_col2, seg_col3 = st.columns(3)

    with seg_col1:
        signal_strength = st.number_input(
            "Signal Strength (dBm)", min_value=-150.0, max_value=0.0, value=-90.0, format="%.2f"
        )

    with seg_col2:
        throughput = st.number_input(
            "Data Throughput (Mbps)", min_value=0.0, max_value=1000.0, value=20.0, format="%.2f"
        )

    with seg_col3:
        latency = st.number_input(
            "Latency (ms)", min_value=0.0, max_value=1000.0, value=100.0, format="%.2f"
        )

    classify_button = st.button("Identify Performance Cluster", use_container_width=True, type="primary")

    if not classify_button:
        st.caption("Results will appear here once a cluster is identified.")

    if classify_button:

        clustering_input = pd.DataFrame({
            "Signal Strength (dBm)": [signal_strength],
            "Data Throughput (Mbps)": [throughput],
            "Latency (ms)": [latency]
        })

        kmeans_model = models["K-Means"]
        cluster = int(kmeans_model.predict(clustering_input)[0])
        performance_level = CLUSTER_LABELS[cluster]

        st.divider()

        section_header("Segmentation Result")

        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            with st.container(border=True):
                st.markdown('<div class="np-card-title">Performance Level</div>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                render_badge(performance_level, performance_tone(performance_level))

        with res_col2:
            with st.container(border=True):
                st.metric("Cluster", f"Cluster {cluster}")

        with res_col3:
            with st.container(border=True):
                st.metric("Throughput", f"{throughput:.2f} Mbps")


# ---------------------------------------------------------------------------
# Model Evaluation Page
# ---------------------------------------------------------------------------

if page == "Model Evaluation":

    st.title("Model Evaluation")
    st.caption(
        "Evaluate a selected regression model by comparing its predictions "
        "against actual throughput values from the test set."
    )

    st.divider()

    model_name = st.selectbox("Regression Model", REGRESSION_MODELS)

    selected_row = model_comparison[model_comparison["Model"] == model_name].iloc[0]

    section_header("Performance Summary")

    eval_col1, eval_col2, eval_col3 = st.columns(3)

    with eval_col1:
        with st.container(border=True):
            st.metric("MAE", f'{selected_row["MAE"]:.4f}')

    with eval_col2:
        with st.container(border=True):
            st.metric("RMSE", f'{selected_row["RMSE"]:.4f}')

    with eval_col3:
        with st.container(border=True):
            st.metric("R² Score", f'{selected_row["R²"]:.4f}')

    st.divider()

    section_header("Actual vs. Predicted Throughput")

    x_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

    selected_model = models[model_name]
    predictions = selected_model.predict(x_test)
    predictions = np.asarray(predictions).ravel()

    evaluation_plot = pd.DataFrame({
        "Actual": y_test,
        "Predicted": predictions
    })

    st.scatter_chart(evaluation_plot, x="Actual", y="Predicted")
    st.caption("Points closer to a diagonal line indicate predictions that closely match actual values.")

    st.divider()

    section_header("Prediction Residuals", "Residual = actual value minus predicted value.")

    residuals = y_test - predictions

    residual_df = pd.DataFrame({
        "Prediction Index": range(len(residuals)),
        "Residual": residuals
    })

    st.line_chart(residual_df, x="Prediction Index", y="Residual")

    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        with st.container(border=True):
            st.metric("Mean Residual", f"{residuals.mean():.4f}")

    with res_col2:
        with st.container(border=True):
            st.metric("Residual Std. Dev.", f"{residuals.std():.4f}")

    with res_col3:
        with st.container(border=True):
            st.metric("Max Absolute Residual", f"{residuals.abs().max():.4f}")

    st.divider()

    section_header("Residual Distribution", "How prediction errors are distributed around zero.")

    hist_counts, bin_edges = np.histogram(residuals, bins=20)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    residual_hist_df = pd.DataFrame({
        "Residual": bin_centers,
        "Count": hist_counts
    })

    st.bar_chart(residual_hist_df, x="Residual", y="Count")
    st.caption(
        "A residual distribution centered closer to zero generally indicates "
        "smaller systematic prediction errors."
    )


# ---------------------------------------------------------------------------
# About Page
# ---------------------------------------------------------------------------

if page == "About":

    st.title("About NetPredict")

    section_header("Project Overview")

    st.markdown(
        """
        NetPredict is a machine learning application developed to analyze
        network performance and predict data throughput based on network
        conditions and geographical characteristics. The project combines
        supervised and unsupervised machine learning techniques to provide
        both throughput prediction and network performance segmentation.
        """
    )

    st.divider()

    section_header("Machine Learning Models")

    model_info = pd.DataFrame({
        "Model": ["Linear Regression", "KNN", "SVR", "Random Forest", "XGBoost", "K-Means"],
        "Type": ["Regression", "Regression", "Regression", "Regression", "Regression", "Clustering"],
        "Purpose": [
            "Baseline regression model",
            "Distance-based prediction",
            "Non-linear regression",
            "Ensemble tree-based regression",
            "Gradient boosting regression",
            "Network performance segmentation"
        ]
    })

    st.dataframe(model_info, use_container_width=True, hide_index=True)

    st.divider()

    section_header("Project Workflow")

    workflow_steps = [
        ("Data Exploration", "Analyze the dataset and identify important patterns, missing values, redundant features, and data quality issues."),
        ("Data Preprocessing", "Clean the dataset, prepare categorical and numerical features, and create consistent training and testing datasets."),
        ("Model Development", "Train multiple regression models to predict network throughput."),
        ("Model Evaluation", "Compare models using MAE, RMSE, and R² Score."),
        ("Network Segmentation", "Apply K-Means clustering to identify different network performance groups."),
        ("Deployment", "Integrate the trained models into an interactive Streamlit application.")
    ]

    wf_col1, wf_col2 = st.columns(2)

    for i, (title, body) in enumerate(workflow_steps):
        target_col = wf_col1 if i % 2 == 0 else wf_col2
        with target_col:
            render_step(i + 1, title, body)

    st.divider()

    section_header("Technologies")

    st.markdown(
        """
        <div class="np-pill-row">
            <span class="np-pill">Python</span>
            <span class="np-pill">Pandas</span>
            <span class="np-pill">NumPy</span>
            <span class="np-pill">Scikit-learn</span>
            <span class="np-pill">XGBoost</span>
            <span class="np-pill">Joblib</span>
            <span class="np-pill">Streamlit</span>
        </div>
        """,
        unsafe_allow_html=True
    )