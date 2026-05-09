from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DATA_DIR = PROJECT_ROOT / "app_data" / "model_results"

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
            "timestamp": pd.to_datetime(df[timestamp_col], errors="coerce", utc=False),
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
    out["timestamp"] = out["timestamp"].dt.tz_localize(None) if getattr(out["timestamp"].dt, "tz", None) else out["timestamp"]
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
    return pd.DataFrame(rows).set_index("Metric")


def style_metric_table(df: pd.DataFrame):
    def highlight_best(row: pd.Series) -> list[str]:
        numeric = pd.to_numeric(row, errors="coerce")
        best = numeric.min(skipna=True)
        return [
            "background-color: #dcfce7; color: #14532d; font-weight: 700" if pd.notna(value) and value == best else ""
            for value in numeric
        ]

    return df.style.format("{:,.2f}", na_rep="n/a").apply(highlight_best, axis=1)


def actual_vs_predicted_plot(df: pd.DataFrame, selected_labels: list[str], granularity: str) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.update_layout(title="Actual vs Predicted Demand")
        return fig

    actual_df = df.groupby("timestamp", as_index=False).agg(actual=("actual", "mean")).sort_values("timestamp")
    fig.add_trace(
        go.Scatter(
            x=actual_df["timestamp"],
            y=actual_df["actual"],
            mode="lines",
            name="Actual Demand",
            line=dict(color=ACTUAL_COLOR, width=ACTUAL_TRACE_WIDTH),
            hovertemplate="Timestamp: %{x}<br>Actual demand: %{y:,.0f} MWh<extra>Actual</extra>",
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
                    "Timestamp: %{x}<br>"
                    "Model: " + label + "<br>"
                    "Actual demand: %{customdata[0]:,.0f} MWh<br>"
                    "Predicted demand: %{y:,.0f} MWh<br>"
                    "Residual: %{customdata[1]:,.0f} MWh<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=f"Actual vs Predicted Demand ({granularity})",
        xaxis_title="Date",
        yaxis_title="Demand (MWh)",
        height=560,
        hovermode="x unified",
        legend_title_text="Series",
        margin=dict(l=10, r=20, t=55, b=40),
    )
    return fig


def residual_plot(df: pd.DataFrame, selected_labels: list[str]) -> go.Figure:
    fig = go.Figure()
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
                hovertemplate="Timestamp: %{x}<br>Model: " + label + "<br>Residual: %{y:,.0f} MWh<extra></extra>",
            )
        )
    fig.add_hline(y=0, line_dash="dash", line_color="#374151")
    fig.update_layout(
        title="Residuals Over Time",
        xaxis_title="Date",
        yaxis_title="Actual - Predicted (MWh)",
        height=440,
        margin=dict(l=10, r=20, t=55, b=40),
    )
    return fig


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
        title=f"{metric} by Forecast Horizon (Full Selected Split)",
    )
    fig.update_traces(opacity=MODEL_TRACE_OPACITY)
    fig.update_layout(height=440, margin=dict(l=10, r=20, t=55, b=40))
    return fig


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
        title="XGBoost Feature Importance / SHAP Summary",
    )
    fig.update_layout(height=620, margin=dict(l=10, r=20, t=55, b=40))
    fig.update_yaxes(categoryorder="array", categoryarray=ordered_features, autorange="reversed")
    return fig


def date_bounds(predictions: list[pd.DataFrame], split: str, selected_labels: list[str]) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    frames = [df[(df["split"] == split) & (df["model_name"].isin(selected_labels))] for df in predictions if not df.empty]
    if not frames:
        return None, None
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        return None, None
    return combined["timestamp"].min(), combined["timestamp"].max()


st.title("⚡ SOCO 48-Hour Energy Demand Forecasting Results")
st.caption(
    "Interactive portfolio dashboard for static SARIMAX, Prophet, full XGBoost, and pruned XGBoost results. "
    "The app reads files from app_data/model_results and does not train models or connect to MLflow."
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

with st.sidebar:
    st.header("View Settings")
    page = st.radio("Page", ["Results Dashboard", "About the App"], index=0)
    st.markdown("**Model Selection**")
    selected_labels = []
    for label in available_labels:
        checked = st.checkbox(label, value=label in default_labels, key=f"model_checkbox_{MODEL_KEYS_BY_LABEL[label]}")
        if checked:
            selected_labels.append(label)
    if not selected_labels:
        st.warning("Select at least one model to display plots.")
    split = st.radio(
        "Evaluation split",
        options=["test", "validation"],
        index=0 if default_split == "test" else 1,
        horizontal=True,
    )
    granularity = st.radio("Prediction Plot Granularity", ["Daily", "Hourly"], index=0, horizontal=True)
    selected_keys = [MODEL_KEYS_BY_LABEL[label] for label in selected_labels]
    all_granular_predictions = {
        key: predictions_for_granularity(data, granularity)
        for key, data in model_data.items()
    }
    selected_prediction_frames = [all_granular_predictions[key] for key in selected_keys if key in all_granular_predictions]
    min_date, max_date = date_bounds(selected_prediction_frames, split, selected_labels)
    if min_date is not None and max_date is not None:
        start, end = st.slider(
            "Displayed date range",
            min_value=min_date.to_pydatetime(),
            max_value=max_date.to_pydatetime(),
            value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
        )
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
    else:
        st.info("No date range is available for the selected models and split.")
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
        st.plotly_chart(actual_vs_predicted_plot(filtered_combined, selected_labels, granularity), width="stretch")

    st.subheader("🎯 Filtered Model Metrics")
    st.caption("Metrics are recalculated from the currently selected split, date range, and plot granularity. Lower values are better; green marks the best model for each row.")
    metrics_df = metric_table(filtered_by_model, available)
    st.dataframe(style_metric_table(metrics_df), width="stretch")

    st.subheader("Residuals")
    st.caption("Residuals near zero indicate lower bias/error. Clusters or spikes may indicate peak demand events, weather extremes, holidays, or other difficult operating conditions.")
    if filtered_combined.empty:
        st.info("Residual data are not available for the current selection.")
    else:
        st.plotly_chart(residual_plot(filtered_combined, selected_labels), width="stretch")

    st.subheader("Error by Forecast Horizon")
    st.caption("This chart uses the full selected validation/test split, not the displayed date range. Forecast error often grows with horizon; unrealistically flat or very low horizon error can be a leakage warning.")
    horizon_chart_col, horizon_control_col = st.columns([4, 1])
    with horizon_control_col:
        horizon_metric = st.radio("Horizon Metric", ["MAE", "RMSE", "MAPE"], index=0)
    horizon_frames = []
    for key in selected_keys:
        horizon_df = model_data[key]["horizon"]
        if not horizon_df.empty:
            horizon_frames.append(horizon_df[horizon_df["split"] == split])
    if horizon_frames:
        horizon_combined = pd.concat(horizon_frames, ignore_index=True)
        with horizon_chart_col:
            st.plotly_chart(horizon_plot(horizon_combined, horizon_metric, selected_labels), width="stretch")
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
            st.plotly_chart(interpretability_plot(pd.concat(importance_frames, ignore_index=True), anchor_key), width="stretch")
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
