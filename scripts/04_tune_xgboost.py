from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import mlflow
import optuna
import pandas as pd
from xgboost import XGBRegressor

from soco_forecasting.config import ensure_artifact_dirs, load_config, project_path
from soco_forecasting.data import (
    create_time_splits,
    get_feature_columns,
    load_modeling_data,
    validate_leakage_safe_feature_names,
)
from soco_forecasting.metrics import prediction_frame, regression_metrics
from soco_forecasting.mlflow_utils import log_artifacts, log_split_manifest, setup_mlflow
from soco_forecasting.plots import actual_vs_predicted, feature_importance_plot, residual_plot, save_plotly_figure


MODEL_NAME = "XGBoost"


def xy(df: pd.DataFrame, feature_columns: list[str], target: str):
    return df[feature_columns], df[target]


def main() -> None:
    config = load_config()
    ensure_artifact_dirs(config)
    setup_mlflow(config)

    df = load_modeling_data(config)
    splits = create_time_splits(df, config)
    target = config["target_column"]
    dt_col = config["datetime_column"]
    feature_columns = get_feature_columns(df, config)
    validate_leakage_safe_feature_names(feature_columns)

    x_train, y_train = xy(splits.train, feature_columns, target)
    x_validation, y_validation = xy(splits.validation, feature_columns, target)
    x_train_validation = pd.concat([x_train, x_validation])
    y_train_validation = pd.concat([y_train, y_validation])
    x_test, y_test = xy(splits.test, feature_columns, target)

    random_state = config["xgboost"]["random_state"]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        model = XGBRegressor(
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
            **params,
        )
        model.fit(x_train, y_train)
        pred = model.predict(x_validation)
        metrics = regression_metrics(y_validation.values, pred)
        with mlflow.start_run(run_name=f"xgboost_trial_{trial.number}", nested=True):
            mlflow.set_tag("model_name", MODEL_NAME)
            mlflow.set_tag("stage", "tuning")
            mlflow.log_params(params)
            mlflow.log_metrics({f"validation_{k}": v for k, v in metrics.items()})
        return metrics["rmse"]

    with mlflow.start_run(run_name="xgboost_optuna") as run:
        mlflow.set_tag("model_name", MODEL_NAME)
        log_split_manifest(splits.manifest)
        mlflow.log_param("n_trials", config["xgboost"]["n_trials"])
        mlflow.log_param("n_features", len(feature_columns))
        mlflow.log_param("tuning_metric", "validation_rmse")

        study = optuna.create_study(direction="minimize", study_name="xgboost_validation_rmse")
        study.optimize(objective, n_trials=config["xgboost"]["n_trials"])
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_validation_rmse", float(study.best_value))

        validation_model = XGBRegressor(
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
            **study.best_params,
        )
        validation_model.fit(x_train, y_train)
        validation_pred = validation_model.predict(x_validation)
        validation_metrics = regression_metrics(y_validation.values, validation_pred, prefix="validation_")
        mlflow.log_metrics(validation_metrics)

        final_model = XGBRegressor(
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
            **study.best_params,
        )
        final_model.fit(x_train_validation, y_train_validation)
        test_pred = final_model.predict(x_test)
        test_metrics = regression_metrics(y_test.values, test_pred, prefix="test_")
        mlflow.log_metrics(test_metrics)

        pred_df = pd.concat(
            [
                prediction_frame(splits.validation[dt_col], y_validation, validation_pred, MODEL_NAME, "validation"),
                prediction_frame(splits.test[dt_col], y_test, test_pred, MODEL_NAME, "test"),
            ]
        )
        pred_path = project_path("reports/metrics/xgboost_predictions.csv")
        pred_df.to_csv(pred_path, index=False)

        trials_path = project_path("reports/metrics/xgboost_optuna_trials.csv")
        study.trials_dataframe().to_csv(trials_path, index=False)

        importance_df = pd.DataFrame({"feature": feature_columns, "importance": final_model.feature_importances_})
        importance_path = project_path("reports/metrics/xgboost_feature_importance.csv")
        importance_df.to_csv(importance_path, index=False)

        model_path = project_path("reports/models/xgboost_model.json")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        final_model.save_model(model_path)

        fig_paths = []
        fig_paths += save_plotly_figure(
            actual_vs_predicted(pred_df[pred_df["split"] == "validation"], "XGBoost Validation: Actual vs Predicted"),
            "reports/figures/xgboost_validation_actual_vs_predicted",
        ).values()
        fig_paths += save_plotly_figure(
            actual_vs_predicted(pred_df[pred_df["split"] == "test"], "XGBoost Test: Actual vs Predicted"),
            "reports/figures/xgboost_test_actual_vs_predicted",
        ).values()
        fig_paths += save_plotly_figure(residual_plot(pred_df, "XGBoost Residuals"), "reports/figures/xgboost_residuals").values()
        fig_paths += save_plotly_figure(
            feature_importance_plot(importance_df, "XGBoost Feature Importance"),
            "reports/figures/xgboost_feature_importance",
        ).values()

        log_artifacts([pred_path, trials_path, importance_path, model_path, *fig_paths])
        print(f"MLflow run_id: {run.info.run_id}")
        print({**validation_metrics, **test_metrics})


if __name__ == "__main__":
    main()
