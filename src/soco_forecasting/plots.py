from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .config import project_path


PLOT_TEMPLATE = "plotly_white"


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


def actual_vs_predicted(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["datetime_utc"], y=df["actual"], mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=df["datetime_utc"], y=df["predicted"], mode="lines", name="Predicted"))
    fig.update_layout(
        title=title,
        xaxis_title="Datetime (UTC)",
        yaxis_title="Demand (MWh)",
        template=PLOT_TEMPLATE,
        legend_title_text="Series",
    )
    return fig


def residual_plot(df: pd.DataFrame, title: str) -> go.Figure:
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
