from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .config import project_path


PLOT_TEMPLATE = "plotly_white"


def add_forecast_origin_markers(fig: go.Figure, df: pd.DataFrame, label: str = "Forecast origin reset") -> go.Figure:
    if "forecast_origin" not in df.columns or df.empty:
        return fig

    origins = pd.to_datetime(df["forecast_origin"].dropna().unique())
    if len(origins) == 0:
        return fig

    first_origin = origins.min()
    for origin in sorted(origins):
        fig.add_vline(
            x=origin,
            line_width=1,
            line_dash="dot",
            line_color="rgba(90, 90, 90, 0.28)",
        )

    fig.add_annotation(
        x=first_origin,
        y=1,
        xref="x",
        yref="paper",
        text=label,
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font={"size": 11, "color": "rgba(60, 60, 60, 0.85)"},
        bgcolor="rgba(255, 255, 255, 0.75)",
        bordercolor="rgba(90, 90, 90, 0.25)",
        borderwidth=1,
    )
    return fig


def save_plotly_figure(fig: go.Figure, output_base: str | Path) -> dict[str, str]:
    base = project_path(output_base)
    base.parent.mkdir(parents=True, exist_ok=True)

    html_path = base.with_suffix(".html")
    json_path = base.with_suffix(".json")
    fig.write_html(html_path, include_plotlyjs="cdn")
    fig.write_json(json_path)

    outputs = {"html": str(html_path), "json": str(json_path)}
    try:
        png_path = base.with_suffix(".png")
        fig.write_image(png_path, scale=2)
        outputs["png"] = str(png_path)
    except Exception:
        # Static export requires kaleido. HTML/JSON artifacts remain available.
        pass
    return outputs


def actual_vs_predicted(
    df: pd.DataFrame,
    title: str,
    show_origin_markers: bool = True,
    origin_marker_label: str = "Forecast origin reset",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["datetime_utc"], y=df["actual"], mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=df["datetime_utc"], y=df["predicted"], mode="lines", name="Predicted"))
    if show_origin_markers:
        add_forecast_origin_markers(fig, df, label=origin_marker_label)
    fig.update_layout(
        title=title,
        xaxis_title="Datetime (UTC)",
        yaxis_title="Demand (MWh)",
        template=PLOT_TEMPLATE,
        legend_title_text="Series",
    )
    return fig


def residual_plot(
    df: pd.DataFrame,
    title: str,
    show_origin_markers: bool = True,
    origin_marker_label: str = "Forecast origin reset",
) -> go.Figure:
    fig = px.scatter(
        df,
        x="datetime_utc",
        y="residual",
        color="split",
        title=title,
        template=PLOT_TEMPLATE,
        labels={"datetime_utc": "Datetime (UTC)", "residual": "Residual (MWh)"},
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    if show_origin_markers:
        add_forecast_origin_markers(fig, df, label=origin_marker_label)
    return fig


def model_comparison_bar(metrics_df: pd.DataFrame, metric: str, title: str) -> go.Figure:
    fig = px.bar(
        metrics_df,
        x="model",
        y=metric,
        color="split",
        barmode="group",
        title=title,
        template=PLOT_TEMPLATE,
        labels={metric: metric.upper(), "model": "Model", "split": "Split"},
    )
    return fig


def horizon_metric_plot(horizon_metrics_df: pd.DataFrame, metric: str, title: str) -> go.Figure:
    fig = px.line(
        horizon_metrics_df,
        x="forecast_horizon_hour",
        y=metric,
        color="split",
        markers=True,
        title=title,
        template=PLOT_TEMPLATE,
        labels={
            "forecast_horizon_hour": "Forecast Horizon Hour",
            metric: metric.upper(),
            "split": "Split",
        },
    )
    fig.update_xaxes(dtick=6)
    return fig


def model_horizon_comparison_plot(horizon_metrics_df: pd.DataFrame, metric: str, title: str) -> go.Figure:
    fig = px.line(
        horizon_metrics_df,
        x="forecast_horizon_hour",
        y=metric,
        color="model",
        markers=True,
        title=title,
        template=PLOT_TEMPLATE,
        labels={
            "forecast_horizon_hour": "Forecast Horizon Hour",
            metric: metric.upper(),
            "model": "Model",
        },
    )
    fig.update_xaxes(dtick=6)
    return fig


def feature_importance_plot(importance_df: pd.DataFrame, title: str, top_n: int = 25) -> go.Figure:
    plot_df = importance_df.sort_values("importance", ascending=False).head(top_n).sort_values("importance")
    fig = px.bar(
        plot_df,
        x="importance",
        y="feature",
        orientation="h",
        title=title,
        template=PLOT_TEMPLATE,
        labels={"importance": "Importance", "feature": "Feature"},
    )
    return fig
