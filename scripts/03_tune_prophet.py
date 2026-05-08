from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import mlflow
import optuna
import pandas as pd
from prophet import Prophet

from soco_forecasting.config import ensure_artifact_dirs, load_config, project_path
from soco_forecasting.data import create_time_splits, load_modeling_data
from soco_forecasting.metrics import prediction_frame, regression_metrics
from soco_forecasting.mlflow_utils import log_artifacts, log_split_manifest, setup_mlflow
from soco_forecasting.plots import actual_vs_predicted, residual_plot, save_plotly_figure


MODEL_NAME = "Prophet"


def to_prophet_frame(df: pd.DataFrame, dt_col: str, target: str) -> pd.DataFrame:
    out = df[[dt_col, target]].copy()
    out[dt_col] = pd.to_datetime(out[dt_col]).dt.tz_convert(None)
    out = out.rename(columns={dt_col: "ds", target: "y"})
    return out


def fit_predict(train_df: pd.DataFrame, predict_df: pd.DataFrame, params: dict, config: dict) -> pd.Series:
    model = Prophet(
        daily_seasonality=config["prophet"]["daily_seasonality"],
        weekly_seasonality=config["prophet"]["weekly_seasonality"],
        yearly_seasonality=config["prophet"]["yearly_seasonality"],
        **params,
    )
    model.fit(train_df)
    future = predict_df[["ds"]].copy()
    forecast = model.predict(future)
    return forecast["yhat"]


def main() -> None:
    config = load_config()
    ensure_artifact_dirs(config)
    setup_mlflow(config)

    df = load_modeling_data(config)
    splits = create_time_splits(df, config)
    target = config["target_column"]
    dt_col = config["datetime_column"]

    train_p = to_prophet_frame(splits.train, dt_col, target)
    validation_p = to_prophet_frame(splits.validation, dt_col, target)
    train_validation_p = pd.concat([train_p, validation_p], ignore_index=True)
    test_p = to_prophet_frame(splits.test, dt_col, target)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "changepoint_prior_scale": trial.suggest_float("changepoint_prior_scale", 0.001, 0.5, log=True),
            "seasonality_prior_scale": trial.suggest_float("seasonality_prior_scale", 0.01, 10.0, log=True),
            "holidays_prior_scale": trial.suggest_float("holidays_prior_scale", 0.01, 10.0, log=True),
            "seasonality_mode": trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"]),
        }
        yhat = fit_predict(train_p, validation_p, params, config)
        metrics = regression_metrics(validation_p["y"].values, yhat.values)
        with mlflow.start_run(run_name=f"prophet_trial_{trial.number}", nested=True):
            mlflow.set_tag("model_name", MODEL_NAME)
            mlflow.set_tag("stage", "tuning")
            mlflow.log_params(params)
            mlflow.log_metrics({f"validation_{k}": v for k, v in metrics.items()})
        return metrics["rmse"]

    with mlflow.start_run(run_name="prophet_optuna") as run:
        mlflow.set_tag("model_name", MODEL_NAME)
        log_split_manifest(splits.manifest)
        mlflow.log_param("n_trials", config["prophet"]["n_trials"])
        mlflow.log_param("tuning_metric", "validation_rmse")

        study = optuna.create_study(direction="minimize", study_name="prophet_validation_rmse")
        study.optimize(objective, n_trials=config["prophet"]["n_trials"])
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_validation_rmse", float(study.best_value))

        validation_pred = fit_predict(train_p, validation_p, study.best_params, config)
        validation_metrics = regression_metrics(validation_p["y"].values, validation_pred.values, prefix="validation_")
        mlflow.log_metrics(validation_metrics)

        test_pred = fit_predict(train_validation_p, test_p, study.best_params, config)
        test_metrics = regression_metrics(test_p["y"].values, test_pred.values, prefix="test_")
        mlflow.log_metrics(test_metrics)

        pred_df = pd.concat(
            [
                prediction_frame(splits.validation[dt_col], validation_p["y"], validation_pred, MODEL_NAME, "validation"),
                prediction_frame(splits.test[dt_col], test_p["y"], test_pred, MODEL_NAME, "test"),
            ]
        )
        pred_path = project_path("reports/metrics/prophet_predictions.csv")
        pred_df.to_csv(pred_path, index=False)

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

        log_artifacts([pred_path, trials_path, *fig_paths])
        print(f"MLflow run_id: {run.info.run_id}")
        print({**validation_metrics, **test_metrics})


if __name__ == "__main__":
    main()
