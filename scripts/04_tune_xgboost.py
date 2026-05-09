from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import mlflow
import optuna
import pandas as pd
from xgboost import XGBRegressor
from xgboost.core import XGBoostError

from soco_forecasting.config import ensure_artifact_dirs, load_config, project_path
from soco_forecasting.data import (
    create_time_splits,
    get_feature_columns,
    load_modeling_data,
    validate_leakage_safe_feature_names,
    validate_numeric_feature_columns,
)
from soco_forecasting.leakage import (
    audit_xgboost_feature_columns,
    save_xgboost_leakage_report,
    validate_xgboost_feature_audit,
)
from soco_forecasting.metrics import metrics_by_forecast_horizon, prediction_frame, regression_metrics
from soco_forecasting.mlflow_utils import log_artifacts, log_split_manifest, setup_mlflow
from soco_forecasting.plots import (
    actual_vs_predicted,
    feature_importance_plot,
    horizon_metric_plot,
    residual_plot,
    save_plotly_figure,
)
from soco_forecasting.recursive import recursive_backtest_48h_windows, recursive_backtest_48h_windows_with_features
from soco_forecasting.shap_utils import generate_xgboost_shap_artifacts


MODEL_NAME = "XGBoost"

# Runtime controls. Override these in configs/modeling_config.json or with
# XGB_RUN_MODE=fast|full for quick debugging versus final portfolio runs.
N_TRIALS_FAST = 10
N_TRIALS_FULL = 50
XGB_N_ESTIMATORS_MAX = 500
EARLY_STOPPING_ROUNDS = 50
SHAP_SAMPLE_SIZE = 5000
RUN_SHAP = True
RUN_FULL_RECURSIVE_EVAL = True
USE_GPU = False


def xy(df: pd.DataFrame, feature_columns: list[str], target: str):
    return df[feature_columns], df[target]


def setting(config: dict, key: str, default):
    value = config.get("xgboost", {}).get(key, default)
    return default if value is None else value


def resolve_runtime_settings(config: dict) -> dict:
    xgb_config = config.get("xgboost", {})
    run_mode = os.environ.get("XGB_RUN_MODE", xgb_config.get("run_mode", "fast")).lower()
    if run_mode not in {"fast", "full"}:
        raise ValueError("XGBoost run_mode must be 'fast' or 'full'.")

    mode_defaults = {
        "run_shap": RUN_SHAP and run_mode == "full",
        "run_full_recursive_eval": RUN_FULL_RECURSIVE_EVAL and run_mode == "full",
        "n_trials": N_TRIALS_FULL if run_mode == "full" else N_TRIALS_FAST,
    }

    return {
        "run_mode": run_mode,
        "n_trials": int(setting(config, "n_trials", mode_defaults["n_trials"])),
        "n_trials_fast": int(setting(config, "n_trials_fast", N_TRIALS_FAST)),
        "n_trials_full": int(setting(config, "n_trials_full", N_TRIALS_FULL)),
        "n_estimators_max": int(setting(config, "n_estimators_max", XGB_N_ESTIMATORS_MAX)),
        "early_stopping_rounds": int(setting(config, "early_stopping_rounds", EARLY_STOPPING_ROUNDS)),
        "shap_sample_size": int(setting(config, "shap_sample_size", SHAP_SAMPLE_SIZE)),
        "run_shap": bool(setting(config, "run_shap", mode_defaults["run_shap"])),
        "run_full_recursive_eval": bool(
            setting(config, "run_full_recursive_eval", mode_defaults["run_full_recursive_eval"])
        ),
        "use_gpu": bool(setting(config, "use_gpu", USE_GPU)),
    }


def base_xgb_params(random_state: int, use_gpu: bool) -> dict:
    return {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": "cuda" if use_gpu else "cpu",
        "random_state": random_state,
        "n_jobs": -1,
    }


def suggest_xgb_params(trial: optuna.Trial, random_state: int, use_gpu: bool, n_estimators_max: int) -> dict:
    params = base_xgb_params(random_state=random_state, use_gpu=use_gpu)
    params.update(
        {
            "n_estimators": trial.suggest_int("n_estimators", 200, n_estimators_max),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
        }
    )
    return params


def fit_with_early_stopping(
    params: dict,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    early_stopping_rounds: int,
) -> XGBRegressor:
    model = XGBRegressor(**params, early_stopping_rounds=early_stopping_rounds)
    try:
        model.fit(x_train, y_train, eval_set=[(x_validation, y_validation)], verbose=False)
    except XGBoostError as error:
        if params.get("device") == "cuda":
            raise RuntimeError(
                "XGBoost GPU training failed with USE_GPU=True. "
                "Set xgboost.use_gpu=false in configs/modeling_config.json or unset GPU mode and rerun."
            ) from error
        raise
    return model


def fit_final_model(params: dict, x_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    model = XGBRegressor(**params)
    model.fit(x_train, y_train, verbose=False)
    return model


def best_iteration_tree_count(model: XGBRegressor, fallback: int) -> int:
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is None:
        return fallback
    return max(1, min(int(best_iteration) + 1, fallback))


def add_direct_48h_window_metadata(predictions: pd.DataFrame, horizon_hours: int) -> pd.DataFrame:
    df = predictions.copy()
    sequence = pd.Series(range(len(df)), index=df.index)
    df["forecast_horizon_hour"] = (sequence % horizon_hours) + 1
    window_id = sequence // horizon_hours
    first_timestamp = df.groupby(window_id)["datetime_utc"].transform("first")
    df["forecast_origin"] = first_timestamp - pd.Timedelta(hours=1)
    return df


def direct_prediction_dataframe(
    model: XGBRegressor,
    df: pd.DataFrame,
    x: pd.DataFrame,
    target: str,
    dt_col: str,
    split_name: str,
    horizon_hours: int,
) -> pd.DataFrame:
    predictions = prediction_frame(
        datetime=df[dt_col],
        actual=df[target],
        predicted=model.predict(x),
        model_name=MODEL_NAME,
        split_name=split_name,
    )
    return add_direct_48h_window_metadata(predictions, horizon_hours)


def log_horizon_metrics(horizon_metrics_df: pd.DataFrame) -> None:
    for split_name in ["validation", "test"]:
        split_horizon_metrics = horizon_metrics_df[horizon_metrics_df["split"] == split_name]
        for horizon_hour in [1, 24, 48]:
            row = split_horizon_metrics[split_horizon_metrics["forecast_horizon_hour"] == horizon_hour]
            if not row.empty:
                mlflow.log_metric(f"{split_name}_horizon_{horizon_hour}_rmse", float(row["rmse"].iloc[0]))


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
    validate_numeric_feature_columns(df, feature_columns)
    leakage_audit = audit_xgboost_feature_columns(df, feature_columns, config)
    validate_xgboost_feature_audit(leakage_audit)
    horizon_hours = config["forecast_horizon_hours"]
    random_state = config["xgboost"]["random_state"]
    runtime = resolve_runtime_settings(config)

    x_train, y_train = xy(splits.train, feature_columns, target)
    x_validation, y_validation = xy(splits.validation, feature_columns, target)
    x_train_validation = pd.concat([x_train, x_validation])
    y_train_validation = pd.concat([y_train, y_validation])

    def objective(trial: optuna.Trial) -> float:
        params = suggest_xgb_params(
            trial=trial,
            random_state=random_state,
            use_gpu=runtime["use_gpu"],
            n_estimators_max=runtime["n_estimators_max"],
        )
        model = fit_with_early_stopping(
            params=params,
            x_train=x_train,
            y_train=y_train,
            x_validation=x_validation,
            y_validation=y_validation,
            early_stopping_rounds=runtime["early_stopping_rounds"],
        )
        predictions = model.predict(x_validation)
        metrics = regression_metrics(y_validation.values, predictions)
        best_trees = best_iteration_tree_count(model, params["n_estimators"])
        trial.set_user_attr("best_iteration_trees", best_trees)
        trial.set_user_attr("validation_rmse", metrics["rmse"])

        with mlflow.start_run(run_name=f"xgboost_trial_{trial.number}", nested=True):
            mlflow.set_tag("model_name", MODEL_NAME)
            mlflow.set_tag("stage", "tuning")
            mlflow.set_tag("evaluation_mode", "direct_validation_for_tuning")
            mlflow.log_params(params)
            mlflow.log_param("early_stopping_rounds", runtime["early_stopping_rounds"])
            mlflow.log_param("best_iteration_trees", best_trees)
            mlflow.log_metrics({f"validation_direct_{k}": v for k, v in metrics.items()})
        return metrics["rmse"]

    with mlflow.start_run(run_name=f"xgboost_optuna_{runtime['run_mode']}") as run:
        mlflow.set_tag("model_name", MODEL_NAME)
        mlflow.set_tag("run_mode", runtime["run_mode"])
        mlflow.set_tag("tuning_evaluation_mode", "direct_validation_for_speed")
        mlflow.set_tag(
            "final_evaluation_mode",
            "recursive_48h_windows" if runtime["run_full_recursive_eval"] else "direct_fast_debug",
        )
        mlflow.set_tag("future_weather_policy", "historical weather rows used as forecast weather for backtesting")
        log_split_manifest(splits.manifest)
        mlflow.log_params(runtime)
        mlflow.log_param("n_features", len(feature_columns))
        mlflow.log_param("forecast_horizon_hours", horizon_hours)
        mlflow.log_param("feature_selection_policy", "all_numeric_engineered_features_except_target_timestamps_and_leakage")
        mlflow.log_param("leakage_audit_passed", leakage_audit["passed"])
        mlflow.log_param("leakage_excluded_suspicious_column_count", len(leakage_audit["excluded_suspicious_columns"]))
        mlflow.log_param("tuning_metric", "validation_direct_rmse")

        study = optuna.create_study(direction="minimize", study_name="xgboost_validation_rmse")
        study.optimize(objective, n_trials=runtime["n_trials"], gc_after_trial=True)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_validation_direct_rmse", float(study.best_value))

        best_params = base_xgb_params(random_state=random_state, use_gpu=runtime["use_gpu"])
        best_params.update(study.best_params)

        validation_model = fit_with_early_stopping(
            params=best_params,
            x_train=x_train,
            y_train=y_train,
            x_validation=x_validation,
            y_validation=y_validation,
            early_stopping_rounds=runtime["early_stopping_rounds"],
        )
        best_tree_count = best_iteration_tree_count(validation_model, best_params["n_estimators"])
        mlflow.log_param("selected_n_estimators_after_early_stopping", best_tree_count)

        final_params = best_params.copy()
        final_params["n_estimators"] = best_tree_count

        if runtime["run_full_recursive_eval"]:
            final_model = fit_final_model(final_params, x_train_validation, y_train_validation)
            mlflow.log_params({f"final_{k}": v for k, v in final_params.items()})
            validation_pred_df = recursive_backtest_48h_windows(
                model=validation_model,
                history_df=splits.train,
                forecast_df=splits.validation,
                feature_columns=feature_columns,
                target_col=target,
                datetime_col=dt_col,
                horizon_hours=horizon_hours,
                model_name=MODEL_NAME,
                split_name="validation",
            )
            test_pred_df, test_feature_state_df = recursive_backtest_48h_windows_with_features(
                model=final_model,
                history_df=pd.concat([splits.train, splits.validation]),
                forecast_df=splits.test,
                feature_columns=feature_columns,
                target_col=target,
                datetime_col=dt_col,
                horizon_hours=horizon_hours,
                model_name=MODEL_NAME,
                split_name="test",
            )
        else:
            final_model = validation_model
            mlflow.set_tag("final_model_scope", "training_split_only_fast_debug")
            validation_pred_df = direct_prediction_dataframe(
                model=validation_model,
                df=splits.validation,
                x=x_validation,
                target=target,
                dt_col=dt_col,
                split_name="validation",
                horizon_hours=horizon_hours,
            )
            mlflow.set_tag(
                "debug_warning",
                "Fast mode uses direct validation predictions only; use full mode for final recursive validation/test metrics.",
            )

        validation_metrics = regression_metrics(
            validation_pred_df["actual"].values,
            validation_pred_df["predicted"].values,
            prefix="validation_",
        )
        mlflow.log_metrics(validation_metrics)

        test_metrics = {}
        if runtime["run_full_recursive_eval"]:
            test_metrics = regression_metrics(
                test_pred_df["actual"].values, test_pred_df["predicted"].values, prefix="test_"
            )
            mlflow.log_metrics(test_metrics)
            pred_df = pd.concat([validation_pred_df, test_pred_df], ignore_index=True)
        else:
            pred_df = validation_pred_df

        pred_df = pred_df.copy()
        pred_df["forecast_timestamp"] = pred_df["datetime_utc"]
        pred_df["horizon_hour"] = pred_df["forecast_horizon_hour"]
        pred_path = project_path("reports/metrics/xgboost_predictions.csv")
        pred_df.to_csv(pred_path, index=False)

        horizon_metrics_df = metrics_by_forecast_horizon(pred_df)
        horizon_metrics_path = project_path("reports/metrics/xgboost_horizon_metrics.csv")
        horizon_metrics_df.to_csv(horizon_metrics_path, index=False)
        log_horizon_metrics(horizon_metrics_df)

        trials_path = project_path("reports/metrics/xgboost_optuna_trials.csv")
        study.trials_dataframe().to_csv(trials_path, index=False)

        leakage_audit_json_path = project_path("reports/metrics/xgboost_leakage_audit.json")
        with leakage_audit_json_path.open("w", encoding="utf-8") as f:
            json.dump(leakage_audit, f, indent=2)

        leakage_columns_path = project_path("reports/metrics/xgboost_excluded_suspicious_columns.csv")
        pd.DataFrame({"column": leakage_audit["excluded_suspicious_columns"]}).to_csv(
            leakage_columns_path, index=False
        )

        leakage_report_path = save_xgboost_leakage_report(
            audit=leakage_audit,
            split_manifest=splits.manifest,
            feature_columns=feature_columns,
        )

        features_path = project_path("reports/metrics/xgboost_features.json")
        with features_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "feature_selection_policy": "all_numeric_engineered_features_except_target_timestamps_and_leakage",
                    "target": target,
                    "n_features": len(feature_columns),
                    "features": feature_columns,
                },
                f,
                indent=2,
            )

        importance_df = pd.DataFrame({"feature": feature_columns, "importance": final_model.feature_importances_})
        importance_path = project_path("reports/metrics/xgboost_feature_importance.csv")
        importance_df.to_csv(importance_path, index=False)

        model_path = project_path("reports/models/xgboost_model.json")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        final_model.save_model(model_path)

        fig_paths = []
        if runtime["run_full_recursive_eval"]:
            recursive_debug_sample_path = project_path("reports/metrics/xgboost_recursive_feature_state_sample.csv")
            test_feature_state_df.head(horizon_hours).to_csv(recursive_debug_sample_path, index=False)
            fig_paths += save_plotly_figure(
                actual_vs_predicted(
                    pred_df[pred_df["split"] == "validation"], "XGBoost Validation: Actual vs Predicted"
                ),
                "reports/figures/xgboost_validation_actual_vs_predicted",
            ).values()
            fig_paths += save_plotly_figure(
                actual_vs_predicted(pred_df[pred_df["split"] == "test"], "XGBoost Test: Actual vs Predicted"),
                "reports/figures/xgboost_test_actual_vs_predicted",
            ).values()
            fig_paths += save_plotly_figure(
                residual_plot(pred_df, "XGBoost Residuals"), "reports/figures/xgboost_residuals"
            ).values()
            fig_paths += save_plotly_figure(
                horizon_metric_plot(horizon_metrics_df, "rmse", "XGBoost RMSE by Forecast Horizon"),
                "reports/figures/xgboost_rmse_by_horizon",
            ).values()
            fig_paths += save_plotly_figure(
                feature_importance_plot(importance_df, "XGBoost Feature Importance"),
                "reports/figures/xgboost_feature_importance",
            ).values()

        shap_artifact_dir = None
        if runtime["run_shap"]:
            if not runtime["run_full_recursive_eval"]:
                mlflow.set_tag("shap_skipped_reason", "run_full_recursive_eval=false")
            else:
                shap_artifacts = generate_xgboost_shap_artifacts(
                    model=final_model,
                    feature_state_df=test_feature_state_df,
                    feature_columns=feature_columns,
                    output_dir="reports/shap/xgboost",
                    sample_size=runtime["shap_sample_size"],
                    random_state=random_state,
                )
                shap_artifact_dir = shap_artifacts["shap_dir"]
                mlflow.log_param("shap_actual_sample_size", shap_artifacts["sample_size"])
                mlflow.log_param("shap_n_features_explained", len(feature_columns))
                mlflow.log_param("shap_dependence_plot_count", shap_artifacts["n_dependence_plots"])

        artifact_paths = [
            pred_path,
            horizon_metrics_path,
            trials_path,
            leakage_audit_json_path,
            leakage_columns_path,
            leakage_report_path,
            features_path,
            importance_path,
            model_path,
            *fig_paths,
        ]
        if runtime["run_full_recursive_eval"]:
            artifact_paths.append(recursive_debug_sample_path)
        log_artifacts(artifact_paths)
        if shap_artifact_dir is not None:
            mlflow.log_artifacts(str(shap_artifact_dir), artifact_path="shap")

        print(f"MLflow run_id: {run.info.run_id}")
        print(f"XGBoost run_mode: {runtime['run_mode']}")
        print(f"Full recursive evaluation: {runtime['run_full_recursive_eval']}")
        print(f"SHAP enabled: {runtime['run_shap']}")
        print({**validation_metrics, **test_metrics})


if __name__ == "__main__":
    main()
