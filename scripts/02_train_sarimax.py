from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import mlflow
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from soco_forecasting.config import ensure_artifact_dirs, load_config, project_path
from soco_forecasting.data import create_time_splits, load_modeling_data
from soco_forecasting.metrics import prediction_frame, regression_metrics
from soco_forecasting.mlflow_utils import log_artifacts, log_split_manifest, setup_mlflow
from soco_forecasting.plots import actual_vs_predicted, residual_plot, save_plotly_figure


MODEL_NAME = "SARIMAX"


def main() -> None:
    config = load_config()
    ensure_artifact_dirs(config)
    setup_mlflow(config)

    df = load_modeling_data(config)
    splits = create_time_splits(df, config)
    target = config["target_column"]
    dt_col = config["datetime_column"]
    order = tuple(config["sarimax"]["order"])
    seasonal_order = tuple(config["sarimax"]["seasonal_order"])

    train_y = splits.train.set_index(dt_col)[target].asfreq("h")
    validation_y = splits.validation.set_index(dt_col)[target].asfreq("h")
    train_validation_y = pd.concat([train_y, validation_y])
    test_y = splits.test.set_index(dt_col)[target].asfreq("h")

    with mlflow.start_run(run_name="sarimax_fixed_order") as run:
        mlflow.set_tag("model_name", MODEL_NAME)
        mlflow.log_param("order", order)
        mlflow.log_param("seasonal_order", seasonal_order)
        log_split_manifest(splits.manifest)

        validation_model = SARIMAX(
            train_y,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        validation_pred = validation_model.forecast(steps=len(validation_y))
        validation_metrics = regression_metrics(validation_y.values, validation_pred.values, prefix="validation_")
        mlflow.log_metrics(validation_metrics)

        final_model = SARIMAX(
            train_validation_y,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        test_pred = final_model.forecast(steps=len(test_y))
        test_metrics = regression_metrics(test_y.values, test_pred.values, prefix="test_")
        mlflow.log_metrics(test_metrics)

        pred_df = pd.concat(
            [
                prediction_frame(validation_y.index, validation_y.values, validation_pred.values, MODEL_NAME, "validation"),
                prediction_frame(test_y.index, test_y.values, test_pred.values, MODEL_NAME, "test"),
            ]
        )
        pred_path = project_path("reports/metrics/sarimax_predictions.csv")
        pred_df.to_csv(pred_path, index=False)

        fig_paths = []
        fig_paths += save_plotly_figure(
            actual_vs_predicted(pred_df[pred_df["split"] == "validation"], "SARIMAX Validation: Actual vs Predicted"),
            "reports/figures/sarimax_validation_actual_vs_predicted",
        ).values()
        fig_paths += save_plotly_figure(
            actual_vs_predicted(pred_df[pred_df["split"] == "test"], "SARIMAX Test: Actual vs Predicted"),
            "reports/figures/sarimax_test_actual_vs_predicted",
        ).values()
        fig_paths += save_plotly_figure(
            residual_plot(pred_df, "SARIMAX Residuals"),
            "reports/figures/sarimax_residuals",
        ).values()

        model_path = project_path("reports/models/sarimax_results.pkl")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        final_model.save(model_path)
        log_artifacts([pred_path, model_path, *fig_paths])

        print(f"MLflow run_id: {run.info.run_id}")
        print({**validation_metrics, **test_metrics})


if __name__ == "__main__":
    main()
