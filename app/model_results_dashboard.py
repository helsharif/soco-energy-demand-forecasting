from __future__ import annotations

import json
import html
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DATA_DIR = PROJECT_ROOT / "app_data" / "model_results"
LOCAL_TIMEZONE = "America/New_York"
LOCAL_TIME_LABEL = "US Eastern"
DAILY_TIME_FORMAT = "%b %d, %Y"
HOURLY_TIME_FORMAT = "%b %d, %Y %I:%M %p"
DAILY_TICK_FORMAT = "%b %d<br>%Y"
HOURLY_TICK_FORMAT = "%b %d<br>%I:%M %p"
SLIDER_TIME_FORMAT = "MMM DD, YYYY"

MODEL_ORDER = ["sarimax", "prophet", "xgboost_full", "xgboost_pruned_top50"]
MODEL_LABELS = {
    "sarimax": "SARIMAX",
    "prophet": "Prophet",
    "xgboost_full": "XGBoost Full",
    "xgboost_pruned_top50": "XGBoost Pruned Top-50",
}
MODEL_KEYS_BY_LABEL = {label: key for key, label in MODEL_LABELS.items()}
MODEL_COLORS = {
    "SARIMAX": "#56B4E9",
    "Prophet": "#E69F00",
    "XGBoost Full": "#009E73",
    "XGBoost Pruned Top-50": "#CC79A7",
}
ACTUAL_COLOR = "#000000"
MODEL_TRACE_OPACITY = 0.85
MODEL_TRACE_WIDTH = 2.4
ACTUAL_TRACE_WIDTH = 3
CHART_FONT_SIZE = 21
CHART_TITLE_FONT_SIZE = 26
CHART_AXIS_FONT_SIZE = 20
CHART_LEGEND_FONT_SIZE = 18

TIMESTAMP_CANDIDATES = [
    "forecast_timestamp",
    "datetime",
    "ds",
    "timestamp",
    "datetime_local",
    "datetime_utc",
    "date",
]
ACTUAL_CANDIDATES = ["actual", "y_true", "demand_imputed_pudl_mwh"]
PREDICTED_CANDIDATES = ["predicted", "y_pred", "prediction", "yhat"]
RESIDUAL_CANDIDATES = ["residual", "error"]
SPLIT_CANDIDATES = ["split", "dataset", "eval_split"]
HORIZON_CANDIDATES = ["horizon_hour", "horizon", "step", "forecast_horizon", "forecast_horizon_hour"]


st.set_page_config(
    page_title="SOCO Energy Demand Forecasting",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    html, body, .stApp {
        font-size: 22px;
    }

    .block-container {
        max-width: 2800px;
        padding-top: 4.2rem;
        padding-left: 4rem;
        padding-right: 4rem;
    }

    [data-testid="stSidebar"] {
        min-width: 430px;
        max-width: 480px;
    }

    [data-testid="stSidebar"] * {
        font-size: 1.08rem;
    }

    h1 {
        font-size: 3.25rem !important;
        line-height: 1.12 !important;
        letter-spacing: 0 !important;
    }

    h2 {
        font-size: 2.35rem !important;
        line-height: 1.2 !important;
        letter-spacing: 0 !important;
        margin-top: 2.2rem !important;
    }

    h3 {
        font-size: 1.7rem !important;
        letter-spacing: 0 !important;
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stCaptionContainer"],
    div[data-testid="stDataFrame"] {
        font-size: 1.08rem !important;
        line-height: 1.55 !important;
    }

    .dashboard-subtitle {
        color: #6b7280;
        font-size: 1.12rem !important;
        line-height: 1.55 !important;
        max-width: 1500px;
        margin-top: -0.35rem;
        margin-bottom: 1.8rem;
    }

    label,
    [data-testid="stRadio"] label,
    [data-testid="stCheckbox"] label {
        font-size: 1.08rem !important;
    }

    .stSlider [data-baseweb="slider"] {
        padding-top: 0.35rem;
        padding-bottom: 0.55rem;
    }

    .metric-table-wrap table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        overflow: hidden;
        background: #ffffff;
    }

    .metric-table-wrap th,
    .metric-table-wrap td {
        text-align: center !important;
        vertical-align: middle !important;
        border-right: 1px solid #e5e7eb;
        border-bottom: 1px solid #e5e7eb;
    }

    .metric-table-wrap th {
        background: #f8fafc;
        color: #4b5563;
        font-weight: 600;
    }

    .metric-table-wrap tr:last-child td {
        border-bottom: 0;
    }

    .metric-table-wrap th:last-child,
    .metric-table-wrap td:last-child {
        border-right: 0;
    }

    .metric-cards-mobile {
        display: none;
    }

    /* The mobile filter menu is duplicated from the sidebar because
       Streamlit's native sidebar toggle is easy to miss on phones. We hide
       the expander on desktop using :has() and show it only at phone/tablet
       widths. Chrome, Edge, Safari, and current mobile browsers support this. */
    div[data-testid="stExpander"]:has(.mobile-controls-marker) {
        display: none;
    }

    .mobile-metric-card {
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        background: #ffffff;
        margin-bottom: 0.85rem;
        overflow: hidden;
    }

    .mobile-metric-title {
        background: #f8fafc;
        color: #374151;
        font-weight: 700;
        padding: 0.75rem 0.9rem;
        text-align: center;
        border-bottom: 1px solid #e5e7eb;
    }

    .mobile-metric-row {
        display: flex;
        justify-content: space-between;
        gap: 0.75rem;
        padding: 0.65rem 0.9rem;
        border-bottom: 1px solid #f1f5f9;
    }

    .mobile-metric-row:last-child {
        border-bottom: 0;
    }

    .mobile-metric-model {
        color: #4b5563;
        font-weight: 600;
    }

    .mobile-metric-value {
        color: #111827;
        font-variant-numeric: tabular-nums;
        text-align: right;
    }

    .mobile-metric-best {
        background: #dcfce7;
    }

    .mobile-metric-best .mobile-metric-value {
        color: #14532d;
        font-weight: 800;
    }

    /* Mobile phones: make the dashboard feel intentionally designed rather than
       like a squeezed desktop canvas. Avoid forcing Streamlit's sidebar width:
       on phones that can create a half-collapsed rail that steals chart space.
       Instead, let Streamlit manage the sidebar overlay and optimize the main
       page, charts, and metric summaries for narrow viewports. */
    @media (max-width: 768px) {
        html, body, .stApp {
            font-size: 16px;
        }

        .block-container {
            max-width: 100vw;
            padding-top: 1.25rem;
            padding-left: 0.85rem;
            padding-right: 0.85rem;
        }

        [data-testid="stSidebar"] {
            display: none !important;
        }

        [data-testid="collapsedControl"] {
            display: none !important;
        }

        div[data-testid="stExpander"]:has(.mobile-controls-marker) {
            display: block;
            margin-bottom: 1rem;
        }

        div[data-testid="stExpander"]:has(.mobile-controls-marker) details {
            border: 1px solid #dbe4ef;
            border-radius: 14px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
            background: #ffffff;
        }

        div[data-testid="stExpander"]:has(.mobile-controls-marker) summary {
            font-size: 1rem !important;
            font-weight: 800 !important;
        }

        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {
            margin-left: 0 !important;
            width: 100vw !important;
            max-width: 100vw !important;
        }

        .stPlotlyChart,
        .stPlotlyChart > div,
        .js-plotly-plot,
        .plot-container,
        .svg-container {
            width: 100% !important;
            max-width: 100% !important;
        }

        h1 {
            font-size: 1.45rem !important;
            line-height: 1.18 !important;
            overflow-wrap: anywhere;
        }

        h2 {
            font-size: 1.12rem !important;
            margin-top: 1.35rem !important;
            overflow-wrap: anywhere;
        }

        h3 {
            font-size: 1.1rem !important;
        }

        .dashboard-subtitle {
            font-size: 0.98rem !important;
            line-height: 1.45 !important;
            margin-bottom: 1.1rem;
        }

        [data-testid="stMarkdownContainer"] p,
        [data-testid="stCaptionContainer"] {
            font-size: 0.95rem !important;
            line-height: 1.45 !important;
        }

        .metric-table-wrap {
            display: none;
        }

        .metric-cards-mobile {
            display: block;
        }

        div[data-testid="stVerticalBlock"]:has(.horizon-metric-marker) {
            text-align: center;
        }

        div[data-testid="stVerticalBlock"]:has(.horizon-metric-marker) [data-testid="stRadio"],
        div[data-testid="stVerticalBlock"]:has(.horizon-metric-marker) [data-testid="stRadio"] > div,
        div[data-testid="stVerticalBlock"]:has(.horizon-metric-marker) [role="radiogroup"] {
            justify-content: center !important;
            text-align: center !important;
            width: 100% !important;
        }

        div[data-testid="stVerticalBlock"]:has(.horizon-metric-marker) [role="radiogroup"] {
            display: flex !important;
            flex-direction: row !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 0.9rem !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }

        div[data-testid="stVerticalBlock"]:has(.horizon-metric-marker) [role="radiogroup"] label {
            margin-left: 0 !important;
            margin-right: 0 !important;
        }

        .horizon-metric-label {
            display: block;
            width: 100%;
            text-align: center;
            font-weight: 600;
            margin-bottom: 0.35rem;
        }

    }

    @media (max-width: 430px) {
        .block-container {
            padding-left: 0.65rem;
            padding-right: 0.65rem;
        }

        h1 {
            font-size: 1.32rem !important;
        }

        h2 {
            font-size: 1.05rem !important;
        }

        .mobile-metric-row {
            flex-direction: column;
            gap: 0.25rem;
            text-align: center;
        }

        .mobile-metric-value {
            text-align: center;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def read_csv(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(csv_path)
    except Exception as exc:
        st.warning(f"Could not read {csv_path.name}: {exc}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def read_json(path: str) -> dict:
    json_path = Path(path)
    if not json_path.exists():
        return {}
    try:
        with json_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        st.warning(f"Could not read {json_path.name}: {exc}")
        return {}


def first_present(columns: pd.Index, candidates: list[str]) -> str | None:
    normalized = {str(col).lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    return None


def to_local_time(values: pd.Series) -> pd.Series:
    """Convert UTC-like dashboard timestamps to naive US Eastern datetimes for plotting/widgets."""
    timestamps = pd.to_datetime(values, errors="coerce", utc=True)
    return timestamps.dt.tz_convert(LOCAL_TIMEZONE).dt.tz_localize(None)


def model_dir(model_key: str) -> Path:
    return DATA_DIR / model_key


def load_manifest() -> dict:
    manifest = read_json(str(DATA_DIR / "manifest.json"))
    if manifest:
        return manifest
    return {
        "models": [
            {"key": key, "display_name": MODEL_LABELS.get(key, key.replace("_", " ").title()), "description": ""}
            for key in MODEL_ORDER
        ]
    }


def available_models(manifest: dict) -> list[dict]:
    manifest_models = {model["key"]: model for model in manifest.get("models", [])}
    models = []
    for key in MODEL_ORDER:
        info = manifest_models.get(key, {"key": key, "display_name": MODEL_LABELS[key]})
        info["display_name"] = MODEL_LABELS.get(key, info.get("display_name", key))
        info["available"] = model_dir(key).exists()
        models.append(info)
    return models


def normalize_predictions(df: pd.DataFrame, model_key: str) -> pd.DataFrame:
    if df.empty:
        return df

    timestamp_col = first_present(df.columns, TIMESTAMP_CANDIDATES)
    actual_col = first_present(df.columns, ACTUAL_CANDIDATES)
    predicted_col = first_present(df.columns, PREDICTED_CANDIDATES)
    split_col = first_present(df.columns, SPLIT_CANDIDATES)
    residual_col = first_present(df.columns, RESIDUAL_CANDIDATES)
    horizon_col = first_present(df.columns, HORIZON_CANDIDATES)

    required = [timestamp_col, actual_col, predicted_col]
    if any(col is None for col in required):
        st.warning(f"{MODEL_LABELS[model_key]} predictions are missing timestamp, actual, or predicted columns.")
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "timestamp": to_local_time(df[timestamp_col]),
            "actual": pd.to_numeric(df[actual_col], errors="coerce"),
            "predicted": pd.to_numeric(df[predicted_col], errors="coerce"),
            "split": df[split_col].astype(str).str.lower() if split_col else "test",
            "model_key": model_key,
            "model_name": MODEL_LABELS[model_key],
        }
    )
    if residual_col:
        out["residual"] = pd.to_numeric(df[residual_col], errors="coerce")
    else:
        out["residual"] = out["actual"] - out["predicted"]
    if horizon_col:
        out["horizon_hour"] = pd.to_numeric(df[horizon_col], errors="coerce")
    out = out.dropna(subset=["timestamp", "actual", "predicted"])
    out["date"] = out["timestamp"].dt.floor("D")
    out["residual"] = out["actual"] - out["predicted"]
    return out


def normalize_horizon(df: pd.DataFrame, model_key: str) -> pd.DataFrame:
    if df.empty:
        return df
    horizon_col = first_present(df.columns, HORIZON_CANDIDATES)
    split_col = first_present(df.columns, SPLIT_CANDIDATES)
    if horizon_col is None:
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "horizon_hour": pd.to_numeric(df[horizon_col], errors="coerce"),
            "split": df[split_col].astype(str).str.lower() if split_col else "test",
            "model_key": model_key,
            "model_name": MODEL_LABELS[model_key],
        }
    )
    for source, target in [("mae", "MAE"), ("rmse", "RMSE"), ("mape", "MAPE"), ("MAE", "MAE"), ("RMSE", "RMSE"), ("MAPE", "MAPE")]:
        if source in df.columns and target not in out.columns:
            out[target] = pd.to_numeric(df[source], errors="coerce")
    return out.dropna(subset=["horizon_hour"])


@st.cache_data(show_spinner=False)
def load_model_data(model_key: str) -> dict:
    folder = model_dir(model_key)
    hourly = normalize_predictions(read_csv(str(folder / "predictions.csv")), model_key)
    daily_file = normalize_predictions(read_csv(str(folder / "daily_predictions.csv")), model_key)
    return {
        "metadata": read_json(str(folder / "metadata.json")),
        "hourly_predictions": hourly,
        "daily_predictions_file": daily_file,
        "horizon": normalize_horizon(read_csv(str(folder / "horizon_errors.csv")), model_key),
        "feature_importance": read_csv(str(folder / "feature_importance.csv")),
        "shap_top_features": read_csv(str(folder / "shap_top_features.csv")),
    }


def daily_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    grouped = (
        df.groupby(["model_key", "model_name", "split", "date"], as_index=False)
        .agg(actual=("actual", "mean"), predicted=("predicted", "mean"))
        .rename(columns={"date": "timestamp"})
    )
    grouped["date"] = grouped["timestamp"]
    grouped["residual"] = grouped["actual"] - grouped["predicted"]
    return grouped


def predictions_for_granularity(model_data: dict, granularity: str) -> pd.DataFrame:
    hourly = model_data["hourly_predictions"]
    if granularity == "Hourly":
        return hourly
    if not hourly.empty:
        return daily_aggregate(hourly)
    return model_data["daily_predictions_file"]


def calculate_metrics(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan}
    actual = pd.to_numeric(df["actual"], errors="coerce").to_numpy(dtype=float)
    predicted = pd.to_numeric(df["predicted"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[valid]
    predicted = predicted[valid]
    if actual.size == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan}
    residual = actual - predicted
    nonzero = actual != 0
    mape = np.nan if not np.any(nonzero) else np.mean(np.abs(residual[nonzero] / actual[nonzero])) * 100
    return {
        "RMSE": float(np.sqrt(np.mean(residual**2))),
        "MAE": float(np.mean(np.abs(residual))),
        "MAPE": float(mape),
    }


def format_metric(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:,.2f}"


def metric_table(filtered_by_model: dict[str, pd.DataFrame], models: list[dict]) -> pd.DataFrame:
    rows = []
    for metric in ["RMSE", "MAE", "MAPE"]:
        row = {"Metric": f"{metric} (%)" if metric == "MAPE" else f"{metric} (MWh)"}
        for model in models:
            row[model["display_name"]] = calculate_metrics(filtered_by_model.get(model["key"], pd.DataFrame()))[metric]
        rows.append(row)
    return pd.DataFrame(rows)


def style_metric_table(df: pd.DataFrame):
    def highlight_best(row: pd.Series) -> list[str]:
        numeric = pd.to_numeric(row.drop(labels=["Metric"], errors="ignore"), errors="coerce")
        best = numeric.min(skipna=True)
        styles = []
        for col, value in row.items():
            if col == "Metric":
                styles.append("font-weight: 600; color: #4b5563;")
            elif pd.notna(value) and value == best:
                styles.append("background-color: #dcfce7; color: #14532d; font-weight: 700;")
            else:
                styles.append("")
        return styles

    return (
        df.style.hide(axis="index")
        .format({col: "{:,.2f}" for col in df.columns if col != "Metric"}, na_rep="n/a")
        .apply(highlight_best, axis=1)
        .set_table_styles(
            [
                {
                    "selector": "td",
                    "props": [
                        ("font-size", "20px"),
                        ("padding", "0.85rem 1rem"),
                        ("text-align", "center"),
                    ],
                },
                {
                    "selector": "th",
                    "props": [
                        ("font-size", "20px"),
                        ("font-weight", "700"),
                        ("padding", "0.85rem 1rem"),
                        ("text-align", "center"),
                    ],
                },
            ]
        )
    )


def mobile_metric_cards(df: pd.DataFrame) -> str:
    cards = []
    for _, row in df.iterrows():
        numeric = pd.to_numeric(row.drop(labels=["Metric"], errors="ignore"), errors="coerce")
        best = numeric.min(skipna=True)
        rows = []
        for model_name, value in numeric.items():
            best_class = " mobile-metric-best" if pd.notna(value) and value == best else ""
            rows.append(
                '<div class="mobile-metric-row{best_class}">'
                '<span class="mobile-metric-model">{model_name}</span>'
                '<span class="mobile-metric-value">{value}</span>'
                "</div>".format(
                    best_class=best_class,
                    model_name=html.escape(str(model_name)),
                    value=html.escape(format_metric(value)),
                )
            )
        cards.append(
            '<div class="mobile-metric-card">'
            '<div class="mobile-metric-title">{metric}</div>'
            "{rows}"
            "</div>".format(metric=html.escape(str(row["Metric"])), rows="".join(rows))
        )
    return f'<div class="metric-cards-mobile">{"".join(cards)}</div>'


def polish_figure(
    fig: go.Figure,
    height: int,
    *,
    legend_y: float = -0.18,
    bottom_margin: int = 115,
    top_margin: int = 55,
    title_size: int | None = None,
) -> go.Figure:
    fig.update_layout(
        height=height,
        font=dict(size=CHART_FONT_SIZE, color="#1f2937"),
        title=dict(font=dict(size=title_size or CHART_TITLE_FONT_SIZE, color="#111827")),
        legend=dict(
            font=dict(size=CHART_LEGEND_FONT_SIZE),
            title_font=dict(size=CHART_LEGEND_FONT_SIZE),
            orientation="h",
            yanchor="top",
            y=legend_y,
            xanchor="left",
            x=0,
        ),
        margin=dict(l=20, r=30, t=top_margin, b=bottom_margin),
    )
    fig.update_xaxes(
        title_font=dict(size=CHART_AXIS_FONT_SIZE),
        tickfont=dict(size=CHART_AXIS_FONT_SIZE - 2),
        gridcolor="#e5e7eb",
        zerolinecolor="#d1d5db",
    )
    fig.update_yaxes(
        title_font=dict(size=CHART_AXIS_FONT_SIZE),
        tickfont=dict(size=CHART_AXIS_FONT_SIZE - 2),
        gridcolor="#e5e7eb",
        zerolinecolor="#d1d5db",
    )
    return fig


def plotly_config() -> dict:
    return {
        "responsive": True,
        "displayModeBar": False,
    }


def display_time_format(granularity: str) -> str:
    return DAILY_TIME_FORMAT if granularity == "Daily" else HOURLY_TIME_FORMAT


def tick_time_format(granularity: str) -> str:
    return DAILY_TICK_FORMAT if granularity == "Daily" else HOURLY_TICK_FORMAT


def actual_vs_predicted_plot(df: pd.DataFrame, selected_labels: list[str], granularity: str) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.update_layout(title="")
        return polish_figure(fig, height=900)

    hover_time_format = display_time_format(granularity)
    actual_df = df.groupby("timestamp", as_index=False).agg(actual=("actual", "mean")).sort_values("timestamp")
    fig.add_trace(
        go.Scatter(
            x=actual_df["timestamp"],
            y=actual_df["actual"],
            mode="lines",
            name="Actual Demand",
            line=dict(color=ACTUAL_COLOR, width=ACTUAL_TRACE_WIDTH),
            hovertemplate=f"{LOCAL_TIME_LABEL}: " + "%{x|" + hover_time_format + "}<br>Actual demand: %{y:,.0f} MWh<extra>Actual</extra>",
        )
    )
    for label in selected_labels:
        model_df = df[df["model_name"] == label].sort_values("timestamp")
        if model_df.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=model_df["timestamp"],
                y=model_df["predicted"],
                mode="lines",
                name=label,
                line=dict(color=MODEL_COLORS[label], width=MODEL_TRACE_WIDTH),
                opacity=MODEL_TRACE_OPACITY,
                customdata=np.stack([model_df["actual"], model_df["residual"]], axis=-1),
                hovertemplate=(
                    f"{LOCAL_TIME_LABEL}: " + "%{x|" + hover_time_format + "}<br>"
                    "Model: " + label + "<br>"
                    "Actual demand: %{customdata[0]:,.0f} MWh<br>"
                    "Predicted demand: %{y:,.0f} MWh<br>"
                    "Residual: %{customdata[1]:,.0f} MWh<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title="",
        xaxis_title=f"Date ({LOCAL_TIME_LABEL})",
        yaxis_title="Demand (MWh)",
        hovermode="x unified",
        legend_title_text="Series",
    )
    fig.update_xaxes(tickformat=tick_time_format(granularity))
    return polish_figure(fig, height=900)


def residual_plot(df: pd.DataFrame, selected_labels: list[str], granularity: str) -> go.Figure:
    fig = go.Figure()
    hover_time_format = display_time_format(granularity)
    for label in selected_labels:
        model_df = df[df["model_name"] == label].sort_values("timestamp")
        if model_df.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=model_df["timestamp"],
                y=model_df["residual"],
                mode="markers",
                name=label,
                marker=dict(color=MODEL_COLORS[label], size=5, opacity=MODEL_TRACE_OPACITY),
                hovertemplate=f"{LOCAL_TIME_LABEL}: " + "%{x|" + hover_time_format + "}<br>Model: " + label + "<br>Residual: %{y:,.0f} MWh<extra></extra>",
            )
        )
    fig.add_hline(y=0, line_dash="dash", line_color="#374151")
    fig.update_layout(
        title="Residuals Over Time",
        xaxis_title=f"Date ({LOCAL_TIME_LABEL})",
        yaxis_title="Actual - Predicted (MWh)",
    )
    fig.update_xaxes(tickformat=tick_time_format(granularity))
    return polish_figure(fig, height=650, legend_y=-0.28, bottom_margin=165)


def horizon_plot(df: pd.DataFrame, metric: str, selected_labels: list[str]) -> go.Figure:
    plot_df = df[df["model_name"].isin(selected_labels)].copy()
    fig = px.line(
        plot_df,
        x="horizon_hour",
        y=metric,
        color="model_name",
        markers=True,
        color_discrete_map=MODEL_COLORS,
        labels={"horizon_hour": "Forecast Horizon Hour", metric: metric, "model_name": "Model"},
        title=f"{metric} by Forecast Horizon",
    )
    fig.update_traces(opacity=MODEL_TRACE_OPACITY)
    return polish_figure(fig, height=650, legend_y=-0.25, bottom_margin=155)


def prepare_importance(model_key: str, data: dict, top_n: int = 20) -> pd.DataFrame:
    shap_df = data["shap_top_features"]
    fi_df = data["feature_importance"]
    score_col = None
    source = None
    if not shap_df.empty and {"feature", "mean_abs_shap"}.issubset(shap_df.columns):
        df = shap_df[["feature", "mean_abs_shap"]].copy()
        score_col = "mean_abs_shap"
        source = "Mean absolute SHAP"
    elif not fi_df.empty:
        feature_col = first_present(fi_df.columns, ["feature", "Feature"])
        importance_col = first_present(fi_df.columns, ["importance", "gain", "weight", "score"])
        if feature_col and importance_col:
            df = fi_df[[feature_col, importance_col]].rename(columns={feature_col: "feature", importance_col: "importance"})
            score_col = "importance"
            source = "Feature importance"
        else:
            return pd.DataFrame()
    else:
        return pd.DataFrame()

    df["score"] = pd.to_numeric(df[score_col], errors="coerce")
    df = df.dropna(subset=["feature", "score"]).sort_values("score", ascending=False).head(top_n)
    df["model_key"] = model_key
    df["model_name"] = MODEL_LABELS[model_key]
    df["source"] = source
    return df


def interpretability_plot(importance_df: pd.DataFrame, anchor_model_key: str | None = None) -> go.Figure:
    if importance_df.empty:
        return go.Figure()
    if anchor_model_key and anchor_model_key in importance_df["model_key"].unique():
        order_source = importance_df[importance_df["model_key"] == anchor_model_key]
    else:
        order_source = importance_df
    ordered_features = order_source.groupby("feature")["score"].max().sort_values(ascending=False).index.tolist()
    fig = px.bar(
        importance_df,
        x="score",
        y="feature",
        color="model_name",
        barmode="group",
        orientation="h",
        category_orders={"feature": ordered_features},
        color_discrete_map=MODEL_COLORS,
        labels={"score": "Importance", "feature": "Feature", "model_name": "Model"},
        title="XGBoost Feature Importance",
    )
    fig.update_yaxes(categoryorder="array", categoryarray=ordered_features, autorange="reversed")
    return polish_figure(fig, height=860, top_margin=70, title_size=CHART_TITLE_FONT_SIZE - 2)


def date_bounds(predictions: list[pd.DataFrame], split: str, selected_labels: list[str]) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    frames = [df[(df["split"] == split) & (df["model_name"].isin(selected_labels))] for df in predictions if not df.empty]
    if not frames:
        return None, None
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        return None, None
    return combined["timestamp"].min(), combined["timestamp"].max()


def set_widget_value(widget_key: str, value) -> None:
    if st.session_state.get(widget_key) != value:
        st.session_state[widget_key] = value


def reset_date_range_state() -> None:
    for key in ["date_range_value", "sidebar_date_range", "mobile_date_range"]:
        if key in st.session_state:
            del st.session_state[key]


def sync_choice(source_key: str, canonical_key: str, peer_keys: list[str]) -> None:
    value = st.session_state[source_key]
    resets_date_range = canonical_key in {"split_choice", "granularity_choice"} or canonical_key.startswith("model_selected_")
    if resets_date_range and st.session_state.get(canonical_key) != value:
        reset_date_range_state()
    st.session_state[canonical_key] = value
    for peer_key in peer_keys:
        set_widget_value(peer_key, value)


def initialize_control_state(available_labels: list[str], default_split: str) -> None:
    st.session_state.setdefault("page_choice", "Results Dashboard")
    st.session_state.setdefault("split_choice", default_split)
    st.session_state.setdefault("granularity_choice", "Daily")
    for label in available_labels:
        model_key = MODEL_KEYS_BY_LABEL[label]
        st.session_state.setdefault(f"model_selected_{model_key}", True)


def selected_labels_from_state(available_labels: list[str]) -> list[str]:
    return [
        label
        for label in available_labels
        if st.session_state.get(f"model_selected_{MODEL_KEYS_BY_LABEL[label]}", True)
    ]


def render_filter_controls(prefix: str, available_labels: list[str], split_options: list[str], include_date: bool = False, date_bounds_value=None) -> tuple[pd.Timestamp | None, pd.Timestamp | None] | None:
    set_widget_value(f"{prefix}_page", st.session_state["page_choice"])
    st.radio(
        "Page",
        ["Results Dashboard", "About the App"],
        key=f"{prefix}_page",
        on_change=sync_choice,
        args=(f"{prefix}_page", "page_choice", ["sidebar_page", "mobile_page"]),
    )

    st.markdown("**Model Selection**")
    for label in available_labels:
        model_key = MODEL_KEYS_BY_LABEL[label]
        canonical_key = f"model_selected_{model_key}"
        widget_key = f"{prefix}_model_{model_key}"
        set_widget_value(widget_key, st.session_state[canonical_key])
        st.checkbox(
            label,
            key=widget_key,
            on_change=sync_choice,
            args=(widget_key, canonical_key, [f"sidebar_model_{model_key}", f"mobile_model_{model_key}"]),
        )

    set_widget_value(f"{prefix}_split", st.session_state["split_choice"])
    st.radio(
        "Evaluation split",
        options=split_options,
        key=f"{prefix}_split",
        horizontal=True,
        on_change=sync_choice,
        args=(f"{prefix}_split", "split_choice", ["sidebar_split", "mobile_split"]),
    )

    set_widget_value(f"{prefix}_granularity", st.session_state["granularity_choice"])
    st.radio(
        "Prediction Plot Granularity",
        ["Daily", "Hourly"],
        key=f"{prefix}_granularity",
        horizontal=True,
        on_change=sync_choice,
        args=(f"{prefix}_granularity", "granularity_choice", ["sidebar_granularity", "mobile_granularity"]),
    )

    if not include_date:
        return None

    min_date, max_date = date_bounds_value
    if min_date is None or max_date is None:
        st.info("No date range is available for the selected models and split.")
        return None, None

    bounded_value = st.session_state.get(
        "date_range_value",
        (min_date.to_pydatetime(), max_date.to_pydatetime()),
    )
    bounded_start = max(pd.Timestamp(bounded_value[0]), min_date).to_pydatetime()
    bounded_end = min(pd.Timestamp(bounded_value[1]), max_date).to_pydatetime()
    if bounded_start > bounded_end:
        bounded_start, bounded_end = min_date.to_pydatetime(), max_date.to_pydatetime()
    st.session_state["date_range_value"] = (bounded_start, bounded_end)
    set_widget_value(f"{prefix}_date_range", st.session_state["date_range_value"])
    st.slider(
        f"Displayed date range ({LOCAL_TIME_LABEL})",
        min_value=min_date.to_pydatetime(),
        max_value=max_date.to_pydatetime(),
        key=f"{prefix}_date_range",
        format=SLIDER_TIME_FORMAT,
        on_change=sync_choice,
        args=(f"{prefix}_date_range", "date_range_value", ["sidebar_date_range", "mobile_date_range"]),
    )
    return pd.Timestamp(st.session_state["date_range_value"][0]), pd.Timestamp(st.session_state["date_range_value"][1])


st.title("⚡ SOCO 48-Hour Energy Demand Forecasting Results")
st.markdown(
    """
    <p class="dashboard-subtitle">
    Interactive portfolio dashboard comparing <strong>SARIMAX</strong>, <strong>Prophet</strong>, and <strong>XGBoost</strong>
    for 48-hour electricity demand forecasting across the Southern Company (SOCO) balancing authority. The project blends
    historical load patterns, weather signals, calendar effects, and engineered time-series features to evaluate forecasting
    performance in a realistic grid-operations setting.
    </p>
    """,
    unsafe_allow_html=True,
)

if not DATA_DIR.exists():
    st.error(f"Dashboard data folder not found: {DATA_DIR}")
    st.stop()

manifest = load_manifest()
models = available_models(manifest)
available = [model for model in models if model["available"]]
if not available:
    st.error("No model result folders are available in app_data/model_results.")
    st.stop()

model_data = {model["key"]: load_model_data(model["key"]) for model in available}
available_labels = [model["display_name"] for model in available]
default_labels = available_labels

split_candidates = sorted(
    {
        split
        for data in model_data.values()
        for split in data["hourly_predictions"].get("split", pd.Series(dtype=str)).dropna().unique().tolist()
    }
)
default_split = "test" if "test" in split_candidates else (split_candidates[0] if split_candidates else "test")
split_options = ["test", "validation"]
initialize_control_state(available_labels, default_split)

page = st.session_state["page_choice"]
split = st.session_state["split_choice"]
granularity = st.session_state["granularity_choice"]
selected_labels = selected_labels_from_state(available_labels)
if not selected_labels:
    st.warning("Select at least one model to display plots.")
selected_keys = [MODEL_KEYS_BY_LABEL[label] for label in selected_labels]
all_granular_predictions = {
    key: predictions_for_granularity(data, granularity)
    for key, data in model_data.items()
}
selected_prediction_frames = [all_granular_predictions[key] for key in selected_keys if key in all_granular_predictions]
min_date, max_date = date_bounds(selected_prediction_frames, split, selected_labels)

with st.sidebar:
    sidebar_date_range = render_filter_controls(
        "sidebar",
        available_labels,
        split_options,
        include_date=True,
        date_bounds_value=(min_date, max_date),
    )

with st.expander("☰ View Settings / Filters", expanded=False):
    st.markdown('<span class="mobile-controls-marker"></span>', unsafe_allow_html=True)
    st.caption("Use these mobile-friendly controls to change page, models, split, granularity, and date range.")
    mobile_date_range = render_filter_controls(
        "mobile",
        available_labels,
        split_options,
        include_date=True,
        date_bounds_value=(min_date, max_date),
    )

date_range = st.session_state.get("date_range_value")
if date_range is not None and min_date is not None and max_date is not None:
    start_ts = pd.Timestamp(date_range[0])
    end_ts = pd.Timestamp(date_range[1])
else:
    start_ts = None
    end_ts = None

if page == "Results Dashboard":
    st.subheader("📈 Actual vs Predicted Time Series")
    if start_ts is None or end_ts is None:
        st.warning("No prediction rows are available for the selected models and split.")
        filtered_combined = pd.DataFrame()
        filtered_by_model = {}
    else:
        filtered_by_model = {}
        for key, pred_df in all_granular_predictions.items():
            model_df = pred_df[pred_df["split"] == split].copy()
            model_df = model_df[(model_df["timestamp"] >= start_ts) & (model_df["timestamp"] <= end_ts)]
            filtered_by_model[key] = model_df
        selected_filtered = [filtered_by_model[key] for key in selected_keys if key in filtered_by_model]
        filtered_combined = pd.concat(selected_filtered, ignore_index=True) if selected_filtered else pd.DataFrame()
        st.plotly_chart(
            actual_vs_predicted_plot(filtered_combined, selected_labels, granularity),
            width="stretch",
            config=plotly_config(),
        )

    st.subheader("🎯 Filtered Model Metrics")
    st.caption("Metrics are recalculated from the currently selected split, date range, and plot granularity. Lower values are better; green marks the best model for each row.")
    metrics_df = metric_table(filtered_by_model, available)
    st.markdown(
        f'<div class="metric-table-wrap">{style_metric_table(metrics_df).to_html()}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(mobile_metric_cards(metrics_df), unsafe_allow_html=True)

    st.subheader("Residuals")
    st.caption("Residuals near zero indicate lower bias/error. Clusters or spikes may indicate peak demand events, weather extremes, holidays, or other difficult operating conditions.")
    if filtered_combined.empty:
        st.info("Residual data are not available for the current selection.")
    else:
        st.plotly_chart(
            residual_plot(filtered_combined, selected_labels, granularity),
            width="stretch",
            config=plotly_config(),
        )

    st.subheader("Error by Forecast Horizon")
    st.caption("This chart uses the full selected validation/test split, not the displayed date range. Forecast error often grows with horizon; unrealistically flat or very low horizon error can be a leakage warning.")
    horizon_chart_col, horizon_control_col = st.columns([4, 1])
    with horizon_control_col:
        st.markdown('<span class="horizon-metric-marker"></span>', unsafe_allow_html=True)
        st.markdown('<span class="horizon-metric-label">Horizon Metric</span>', unsafe_allow_html=True)
        metric_left, metric_center, metric_right = st.columns([1, 3, 1])
        with metric_center:
            horizon_metric = st.radio(
                "Horizon Metric",
                ["MAE", "RMSE", "MAPE"],
                index=0,
                horizontal=True,
                label_visibility="collapsed",
            )
    horizon_frames = []
    for key in selected_keys:
        horizon_df = model_data[key]["horizon"]
        if not horizon_df.empty:
            horizon_frames.append(horizon_df[horizon_df["split"] == split])
    if horizon_frames:
        horizon_combined = pd.concat(horizon_frames, ignore_index=True)
        with horizon_chart_col:
            st.plotly_chart(
                horizon_plot(horizon_combined, horizon_metric, selected_labels),
                width="stretch",
                config=plotly_config(),
            )
    else:
        st.info("Horizon-level error data are not available for the selected models.")

    xgb_keys = [key for key in selected_keys if key in {"xgboost_full", "xgboost_pruned_top50"}]
    if xgb_keys:
        st.subheader("💡 Interpretability")
        st.caption(
            "Feature importance and SHAP help identify which inputs drive XGBoost forecasts. "
            "Lagged demand, calendar features, CDH, HDH, humidity, and weather features may explain model behavior; the pruned model tests whether a smaller feature set preserves performance."
        )
        importance_frames = [prepare_importance(key, model_data[key], top_n=20) for key in xgb_keys]
        importance_frames = [df for df in importance_frames if not df.empty]
        if importance_frames:
            anchor_key = "xgboost_pruned_top50" if "xgboost_pruned_top50" in xgb_keys else xgb_keys[0]
            st.plotly_chart(
                interpretability_plot(pd.concat(importance_frames, ignore_index=True), anchor_key),
                width="stretch",
                config=plotly_config(),
            )
        else:
            st.info("XGBoost interpretability files are not available for the selected model(s).")

else:
    st.header("ℹ️ About the App")
    st.markdown(
        """
        This dashboard visualizes completed, static MLflow model results for 48-hour-ahead SOCO hourly electricity demand forecasting.
        It does not train, tune, run predictions, compute SHAP, or connect to MLflow at runtime.

        **Models shown**

        * **SARIMAX:** demand-only statistical baseline.
        * **Prophet:** interpretable trend/seasonality model with CDH, HDH, and humidity regressors.
        * **XGBoost Full:** full engineered feature set for nonlinear demand, calendar, weather, and persistence effects.
        * **XGBoost Pruned Top-50:** simplified XGBoost model using the top 50 selected features.

        **How to read the results**

        All dashboard timestamps are displayed in US Eastern local time.
        XGBoost models are expected to capture nonlinear demand persistence, calendar effects, and weather/degree-hour relationships.
        Prophet provides an interpretable benchmark, while SARIMAX provides a demand-only baseline.
        The pruned XGBoost experiment tests whether model simplicity can improve without a major accuracy penalty.

        **Evaluation safeguards**

        The modeling workflow uses time-aware train/validation/test splits and recursive 48-hour evaluation.
        Future actual demand should not be used inside a forecast window, and horizon-by-horizon errors help diagnose 48-hour forecast behavior and leakage risk.

        **Limitations**

        The dashboard uses static exported results from `app_data/model_results/`.
        Future weather in historical backtests may use recorded weather as a proxy for forecast-available weather, depending on the experiment setup.
        Results depend on completed MLflow experiments and exported artifacts.

        **Disclaimer**

        This project is for research, learning, and portfolio demonstration purposes only.
        Forecasts and analysis are illustrative and should not be used for operational utility planning without additional validation.

        **Author**

        Husayn El Sharif  
        Senior Data Scientist / Machine Learning Engineer  
        Portfolio: [https://helsharif.github.io](https://helsharif.github.io)  
        GitHub Repo: [https://github.com/helsharif](https://github.com/helsharif)
        """
    )
