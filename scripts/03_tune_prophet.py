from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import mlflow
import optuna
import pandas as pd

from soco_forecasting.config import ensure_artifact_dirs, load_config, project_path
from soco_forecasting.data import create_time_splits, load_modeling_data
from soco_forecasting.metrics import metrics_by_forecast_horizon, regression_metrics
from soco_forecasting.mlflow_utils import log_artifacts, log_split_manifest, setup_mlflow
from soco_forecasting.plots import actual_vs_predicted, horizon_metric_plot, residual_plot, save_plotly_figure
from soco_forecasting.prophet_model import direct_prophet_48h_windows


MODEL_NAME = "Prophet"


def main() -> None:
    config = load_config()
    ensure_artifact_dirs(config)
    setup_mlflow(config)

    df = load_modeling_data(config)
    splits = create_time_splits(df, config)
    target = config["target_column"]
    dt_col = config["datetime_column"]
    horizon_hours = config["forecast_horizon_hours"]
    tuning_max_windows = config["prophet"].get("tuning_max_windows")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "changepoint_prior_scale": trial.suggest_float("changepoint_prior_scale", 0.001, 0.5, log=True),
            "seasonality_prior_scale": trial.suggest_float("seasonality_prior_scale", 0.01, 10.0, log=True),
            "holidays_prior_scale": trial.suggest_float("holidays_prior_scale", 0.01, 10.0, log=True),
            "seasonality_mode": trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"]),
        }
        pred_df = direct_prophet_48h_windows(
            train_df=splits.train,
            forecast_df=splits.validation,
            params=params,
            config=config,
            target_col=target,
            datetime_col=dt_col,
            horizon_hours=horizon_hours,
            model_name=MODEL_NAME,
            split_name="validation",
            max_windows=tuning_max_windows,
        )
        metrics = regression_metrics(pred_df["actual"].values, pred_df["predicted"].values)
        with mlflow.start_run(run_name=f"prophet_trial_{trial.number}", nested=True):
            mlflow.set_tag("model_name", MODEL_NAME)
            mlflow.set_tag("stage", "tuning")
            mlflow.set_tag("forecast_strategy", "direct_48h_window_evaluation")
            mlflow.log_params(params)
            mlflow.log_param("forecast_horizon_hours", horizon_hours)
            if tuning_max_windows is not None:
                mlflow.log_param("tuning_max_windows", tuning_max_windows)
            mlflow.log_metrics({f"validation_{k}": v for k, v in metrics.items()})
        return metrics["rmse"]

    with mlflow.start_run(run_name="prophet_optuna") as run:
        mlflow.set_tag("model_name", MODEL_NAME)
        log_split_manifest(splits.manifest)
        mlflow.log_param("n_trials", config["prophet"]["n_trials"])
        mlflow.log_param("forecast_horizon_hours", horizon_hours)
        mlflow.set_tag("forecast_strategy", "direct_48h_window_evaluation")
        if tuning_max_windows is not None:
            mlflow.log_param("tuning_max_windows", tuning_max_windows)
        mlflow.log_param("tuning_metric", "validation_rmse")

        study = optuna.create_study(direction="minimize", study_name="prophet_validation_rmse")
        study.optimize(objective, n_trials=config["prophet"]["n_trials"])
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_validation_rmse", float(study.best_value))

        validation_pred_df = direct_prophet_48h_windows(
            train_df=splits.train,
            forecast_df=splits.validation,
            params=study.best_params,
            config=config,
            target_col=target,
            datetime_col=dt_col,
            horizon_hours=horizon_hours,
            model_name=MODEL_NAME,
            split_name="validation",
        )
        validation_metrics = regression_metrics(
            validation_pred_df["actual"].values,
            validation_pred_df["predicted"].values,
            prefix="validation_",
        )
        mlflow.log_metrics(validation_metrics)

        test_pred_df = direct_prophet_48h_windows(
            train_df=pd.concat([splits.train, splits.validation]),
            forecast_df=splits.test,
            params=study.best_params,
            config=config,
            target_col=target,
            datetime_col=dt_col,
            horizon_hours=horizon_hours,
            model_name=MODEL_NAME,
            split_name="test",
        )
        test_metrics = regression_metrics(
            test_pred_df["actual"].values,
            test_pred_df["predicted"].values,
            prefix="test_",
        )
        mlflow.log_metrics(test_metrics)

        pred_df = pd.concat([validation_pred_df, test_pred_df], ignore_index=True)
        pred_path = project_path("reports/metrics/prophet_predictions.csv")
        pred_df.to_csv(pred_path, index=False)

        horizon_metrics_df = metrics_by_forecast_horizon(pred_df)
        horizon_metrics_path = project_path("reports/metrics/prophet_horizon_metrics.csv")
        horizon_metrics_df.to_csv(horizon_metrics_path, index=False)
        for split_name in ["validation", "test"]:
            split_horizon_metrics = horizon_metrics_df[horizon_metrics_df["split"] == split_name]
            for horizon_hour in [1, 24, 48]:
                row = split_horizon_metrics[split_horizon_metrics["forecast_horizon_hour"] == horizon_hour]
                if not row.empty:
                    mlflow.log_metric(f"{split_name}_horizon_{horizon_hour}_rmse", float(row["rmse"].iloc[0]))

        trials_path = project_path("reports/metrics/prophet_optuna_trials.csv")
        study.trials_dataframe().to_csv(trials_path, index=False)

        fig_paths = []
        fig_paths += save_plotly_figure(
            actual_vs_predicted(pred_df[pred_df["split"] == "validation"], "Prophet Validation: Actual vs Predicted"),
            "reports/figures/prophet_validation_actual_vs_predicted",
        ).values()
        fig_paths += save_plotly_figure(
            actual_vs_predicted(pred_df[pred_df["split"] == "test"], "Prophet Test: Actual vs Predicted"),
            "reports/figures/prophet_test_actual_vs_predicted",
        ).values()
        fig_paths += save_plotly_figure(residual_plot(pred_df, "Prophet Residuals"), "reports/figures/prophet_residuals").values()
        fig_paths += save_plotly_figure(
            horizon_metric_plot(horizon_metrics_df, "rmse", "Prophet RMSE by Forecast Horizon"),
            "reports/figures/prophet_rmse_by_horizon",
        ).values()

        log_artifacts([pred_path, horizon_metrics_path, trials_path, *fig_paths])
        print(f"MLflow run_id: {run.info.run_id}")
        print({**validation_metrics, **test_metrics})


if __name__ == "__main__":
    main()
