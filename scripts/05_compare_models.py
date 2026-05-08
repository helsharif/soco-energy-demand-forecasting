from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from soco_forecasting.config import ensure_artifact_dirs, load_config, project_path
from soco_forecasting.metrics import regression_metrics
from soco_forecasting.plots import model_comparison_bar, save_plotly_figure


PREDICTION_FILES = [
    project_path("reports/metrics/sarimax_predictions.csv"),
    project_path("reports/metrics/prophet_predictions.csv"),
    project_path("reports/metrics/xgboost_predictions.csv"),
]


def main() -> None:
    config = load_config()
    ensure_artifact_dirs(config)

    existing = [path for path in PREDICTION_FILES if path.exists()]
    if not existing:
        raise FileNotFoundError("No prediction files found. Run one or more model scripts first.")

    predictions = pd.concat([pd.read_csv(path, parse_dates=["datetime_utc"]) for path in existing], ignore_index=True)
    rows = []
    for (model, split), part in predictions.groupby(["model", "split"]):
        row = {"model": model, "split": split}
        row.update(regression_metrics(part["actual"], part["predicted"]))
        rows.append(row)

    metrics_df = pd.DataFrame(rows).sort_values(["split", "rmse"])
    metrics_path = project_path("reports/metrics/model_comparison.csv")
    metrics_df.to_csv(metrics_path, index=False)

    for metric in ["mae", "rmse", "mape"]:
        save_plotly_figure(
            model_comparison_bar(metrics_df, metric, f"Model Comparison: {metric.upper()}"),
            f"reports/figures/model_comparison_{metric}",
        )

    print(f"Saved comparison table: {metrics_path}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
