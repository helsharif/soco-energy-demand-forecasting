from __future__ import annotations

import json
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
from soco_forecasting.leakage import audit_xgboost_feature_columns, validate_xgboost_feature_audit
from soco_forecasting.metrics import metrics_by_forecast_horizon, regression_metrics
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


MODEL_NAME = "xgboost_pruned_top50"
FAST_MODE = True
N_TRIALS_FAST = 10
N_TRIALS_FULL = 50
XGB_N_ESTIMATORS_MAX = 500
EARLY_STOPPING_ROUNDS = 50
SHAP_SAMPLE_SIZE = 5000
RUN_SHAP = True
RUN_FULL_RECURSIVE_EVAL = True
USE_GPU = False
TOP_N_FEATURES = 50

FEATURE_IMPORTANCE_CANDIDATES = (
    "reports/shap/xgboost/shap_top_features.csv",
    "reports/metrics/xgboost_feature_importance.csv",
)


def xy(df: pd.DataFrame, feature_columns: list[str], target: str):
    return df[feature_columns], df[target]


def base_xgb_params(random_state: int) -> dict:
    return {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": "cuda" if USE_GPU else "cpu",
        "random_state": random_state,
        "n_jobs": -1,
    }


def suggest_xgb_params(trial: optuna.Trial, random_state: int) -> dict:
    params = base_xgb_params(random_state)
    params.update(
        {
            "n_estimators": trial.suggest_int("n_estimators", 200, XGB_N_ESTIMATORS_MAX),
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


def fit_with_early_stopping(params, x_train, y_train, x_validation, y_validation) -> XGBRegressor:
    model = XGBRegressor(**params, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    try:
        model.fit(x_train, y_train, eval_set=[(x_validation, y_validation)], verbose=False)
    except XGBoostError as error:
        if params.get("device") == "cuda":
            raise RuntimeError(
                "XGBoost GPU training failed with USE_GPU=True. Set USE_GPU=False in "
                "scripts/05_tune_xgboost_pruned.py and rerun."
            ) from error
        raise
    return model


def fit_final_model(params, x_train, y_train) -> XGBRegressor:
    model = XGBRegressor(**params)
    model.fit(x_train, y_train, verbose=False)
    return model


def best_iteration_tree_count(model: XGBRegressor, fallback: int) -> int:
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is None:
        return fallback
    return max(1, min(int(best_iteration) + 1, fallback))


def load_ranked_features() -> tuple[pd.DataFrame, Path, str]:
    for candidate in FEATURE_IMPORTANCE_CANDIDATES:
        path = project_path(candidate)
        if not path.exists():
            continue
        importance_df = pd.read_csv(path)
        if "feature" not in importance_df.columns:
            continue
        if "mean_abs_shap" in importance_df.columns:
            score_col = "mean_abs_shap"
        elif "importance" in importance_df.columns:
            score_col = "importance"
        else:
            continue
        ranked = importance_df[["feature", score_col]].copy()
        ranked = ranked.sort_values(score_col, ascending=False).reset_index(drop=True)
        return ranked, path, score_col

    candidates = "\n".join(f"- {path}" for path in FEATURE_IMPORTANCE_CANDIDATES)
    raise FileNotFoundError(
        "No usable full XGBoost feature-importance CSV was found. "
        "Run the full XGBoost experiment first, then rerun this pruned experiment. "
        f"Checked:\n{candidates}"
    )


def select_top_valid_features(
    ranked_features: pd.DataFrame,
    full_valid_features: list[str],
    df: pd.DataFrame,
    config: dict,
) -> tuple[list[str], list[str]]:
    valid_set = set(full_valid_features)
    selected: list[str] = []
    skipped: list[str] = []

    for feature in ranked_features["feature"]:
        if feature in selected:
            continue
        if feature not in valid_set:
            skipped.append(feature)
            continue
        candidate = [*selected, feature]
        audit = audit_xgboost_feature_columns(df, candidate, config)
        if not audit["passed"]:
            skipped.append(feature)
            continue
        selected.append(feature)
        if len(selected) == TOP_N_FEATURES:
            break

    if len(selected) < TOP_N_FEATURES:
        raise ValueError(
            f"Only {len(selected)} leakage-safe ranked features were available; "
            f"{TOP_N_FEATURES} are required. Run the full XGBoost experiment first "
            "and confirm its feature-importance artifact is complete."
        )
    return selected, skipped


def log_horizon_metrics(horizon_metrics_df: pd.DataFrame) -> None:
    for split_name in ["validation", "test"]:
        split_horizon_metrics = horizon_metrics_df[horizon_metrics_df["split"] == split_name]
        for horizon_hour in [1, 24, 48]:
            row = split_horizon_metrics[split_horizon_metrics["forecast_horizon_hour"] == horizon_hour]
            if not row.empty:
                mlflow.log_metric(f"{split_name}_horizon_{horizon_hour}_rmse", float(row["rmse"].iloc[0]))


def maybe_full_metrics_row() -> dict | None:
    path = project_path("reports/metrics/xgboost_predictions.csv")
    if not path.exists():
        return None
    pred_df = pd.read_csv(path)
    required_cols = {"split", "actual", "predicted"}
    if not required_cols.issubset(pred_df.columns):
        return None
    rows = {"model_name": "xgboost_full", "top_n_features": "full"}
    for split_name in ["validation", "test"]:
        part = pred_df[pred_df["split"] == split_name]
        if part.empty:
            continue
        metrics = regression_metrics(part["actual"], part["predicted"])
        rows[f"{split_name}_MAE"] = metrics["mae"]
        rows[f"{split_name}_RMSE"] = metrics["rmse"]
        rows[f"{split_name}_MAPE"] = metrics["mape"]
    rows.update(
        {
            "peak_error_metric": "",
            "best_params_path": "",
            "best_params_summary": "See full XGBoost MLflow run.",
            "feature_list_path": "reports/metrics/xgboost_features.json",
            "mlflow_run_id": "",
        }
    )
    return rows


def save_comparison_and_summary(pruned_row: dict, full_row: dict | None) -> tuple[Path, Path]:
    rows = [row for row in [full_row, pruned_row] if row is not None]
    comparison_df = pd.DataFrame(rows)
    comparison_path = project_path("reports/xgboost_pruned_comparison.csv")
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(comparison_path, index=False)

    recommendation = "Keep the full-feature XGBoost model as the primary model until pruned performance is reviewed."
    if full_row and "test_RMSE" in full_row and "test_RMSE" in pruned_row:
        rmse_ratio = pruned_row["test_RMSE"] / full_row["test_RMSE"]
        if 0.97 <= rmse_ratio <= 1.03:
            recommendation = "The Top-50 pruned model is within about 3% of the full model by test RMSE and is worth considering for promotion."
        else:
            recommendation = "Keep the full-feature XGBoost model as the primary model; the Top-50 run is best treated as an ablation."

    summary_path = project_path("reports/xgboost_pruned_experiment_summary.md")
    summary = f"""# XGBoost Pruned Top-50 Experiment

## Purpose

This experiment tests whether a simpler XGBoost model using only the top {TOP_N_FEATURES} features from the full XGBoost feature-importance results can preserve most of the recursive 48-hour forecasting performance.

## Feature Selection

Features were selected from the full XGBoost importance artifact, preferring `reports/shap/xgboost/shap_top_features.csv`. Ranked features were filtered through the same leakage audit used by the full XGBoost workflow. Unsafe or unavailable features were skipped.

## Evaluation

The pruned model was tuned independently with Optuna using lightweight direct validation predictions. After hyperparameter selection, it was evaluated once with the same leakage-free recursive 48-hour validation/test procedure used by the full XGBoost model.

## Result

- Validation RMSE: {pruned_row.get("validation_RMSE", "")}
- Test RMSE: {pruned_row.get("test_RMSE", "")}
- MLflow run id: {pruned_row.get("mlflow_run_id", "")}

## Recommendation

{recommendation}
"""
    summary_path.write_text(summary, encoding="utf-8")
    return comparison_path, summary_path


def main() -> None:
    config = load_config()
    ensure_artifact_dirs(config)
    setup_mlflow(config)

    df = load_modeling_data(config)
    splits = create_time_splits(df, config)
    target = config["target_column"]
    dt_col = config["datetime_column"]
    horizon_hours = config["forecast_horizon_hours"]
    random_state = config["xgboost"]["random_state"]

    full_valid_features = get_feature_columns(df, config)
    validate_leakage_safe_feature_names(full_valid_features)
    validate_numeric_feature_columns(df, full_valid_features)
    full_audit = audit_xgboost_feature_columns(df, full_valid_features, config)
    validate_xgboost_feature_audit(full_audit)

    ranked_features, source_path, score_col = load_ranked_features()
    feature_columns, skipped_features = select_top_valid_features(ranked_features, full_valid_features, df, config)
    validate_leakage_safe_feature_names(feature_columns)
    validate_numeric_feature_columns(df, feature_columns)
    pruned_audit = audit_xgboost_feature_columns(df, feature_columns, config)
    validate_xgboost_feature_audit(pruned_audit)

    x_train, y_train = xy(splits.train, feature_columns, target)
    x_validation, y_validation = xy(splits.validation, feature_columns, target)
    x_train_validation = pd.concat([x_train, x_validation])
    y_train_validation = pd.concat([y_train, y_validation])
    n_trials = N_TRIALS_FAST if FAST_MODE else N_TRIALS_FULL

    def objective(trial: optuna.Trial) -> float:
        params = suggest_xgb_params(trial, random_state=random_state)
        model = fit_with_early_stopping(params, x_train, y_train, x_validation, y_validation)
        predictions = model.predict(x_validation)
        metrics = regression_metrics(y_validation.values, predictions)
        with mlflow.start_run(run_name=f"{MODEL_NAME}_trial_{trial.number}", nested=True):
            mlflow.set_tag("model_name", MODEL_NAME)
            mlflow.set_tag("stage", "tuning")
            mlflow.set_tag("evaluation_mode", "direct_validation_for_tuning")
            mlflow.log_params(params)
            mlflow.log_param("early_stopping_rounds", EARLY_STOPPING_ROUNDS)
            mlflow.log_metrics({f"validation_direct_{key}": value for key, value in metrics.items()})
        return metrics["rmse"]

    with mlflow.start_run(run_name=f"{MODEL_NAME}_optuna") as run:
        mlflow.set_tag("model_name", MODEL_NAME)
        mlflow.set_tag("run_type", "pruned_feature_ablation")
        mlflow.set_tag("final_evaluation_mode", "recursive_48h_windows")
        mlflow.set_tag("future_weather_policy", "historical weather rows used as forecast weather for backtesting")
        log_split_manifest(splits.manifest)
        mlflow.log_param("fast_mode", FAST_MODE)
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("top_n_features", TOP_N_FEATURES)
        mlflow.log_param("n_selected_features", len(feature_columns))
        mlflow.log_param("feature_importance_source", str(source_path))
        mlflow.log_param("feature_importance_score_column", score_col)
        mlflow.log_param("leakage_audit_passed", pruned_audit["passed"])
        mlflow.log_param("run_shap", RUN_SHAP)
        mlflow.log_param("run_full_recursive_eval", RUN_FULL_RECURSIVE_EVAL)
        mlflow.log_param("use_gpu", USE_GPU)

        study = optuna.create_study(direction="minimize", study_name=f"{MODEL_NAME}_validation_rmse")
        study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
        mlflow.log_params({f"best_{key}": value for key, value in study.best_params.items()})
        mlflow.log_metric("best_validation_direct_rmse", float(study.best_value))

        best_params = base_xgb_params(random_state=random_state)
        best_params.update(study.best_params)
        validation_model = fit_with_early_stopping(best_params, x_train, y_train, x_validation, y_validation)
        best_tree_count = best_iteration_tree_count(validation_model, best_params["n_estimators"])
        final_params = best_params.copy()
        final_params["n_estimators"] = best_tree_count
        final_model = fit_final_model(final_params, x_train_validation, y_train_validation)
        mlflow.log_param("selected_n_estimators_after_early_stopping", best_tree_count)
        mlflow.log_params({f"final_{key}": value for key, value in final_params.items()})

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

        validation_metrics = regression_metrics(
            validation_pred_df["actual"].values,
            validation_pred_df["predicted"].values,
            prefix="validation_",
        )
        test_metrics = regression_metrics(test_pred_df["actual"].values, test_pred_df["predicted"].values, prefix="test_")
        mlflow.log_metrics(validation_metrics)
        mlflow.log_metrics(test_metrics)

        pred_df = pd.concat([validation_pred_df, test_pred_df], ignore_index=True)
        pred_df["forecast_timestamp"] = pred_df["datetime_utc"]
        pred_df["horizon_hour"] = pred_df["forecast_horizon_hour"]
        pred_path = project_path("reports/metrics/xgboost_pruned_top50_predictions.csv")
        pred_df.to_csv(pred_path, index=False)

        horizon_metrics_df = metrics_by_forecast_horizon(pred_df)
        horizon_metrics_path = project_path("reports/metrics/xgboost_pruned_top50_horizon_metrics.csv")
        horizon_metrics_df.to_csv(horizon_metrics_path, index=False)
        log_horizon_metrics(horizon_metrics_df)

        trials_path = project_path("reports/metrics/xgboost_pruned_top50_optuna_trials.csv")
        study.trials_dataframe().to_csv(trials_path, index=False)

        best_params_path = project_path("reports/metrics/xgboost_pruned_top50_best_params.json")
        with best_params_path.open("w", encoding="utf-8") as f:
            json.dump(final_params, f, indent=2)

        features_path = project_path("reports/metrics/xgboost_pruned_top50_features.json")
        with features_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_name": MODEL_NAME,
                    "top_n_features": TOP_N_FEATURES,
                    "feature_importance_source": str(source_path),
                    "score_column": score_col,
                    "target": target,
                    "n_features": len(feature_columns),
                    "features": feature_columns,
                    "skipped_ranked_features": skipped_features,
                },
                f,
                indent=2,
            )

        selected_features_csv_path = project_path("reports/metrics/xgboost_pruned_top50_features.csv")
        pd.DataFrame({"feature": feature_columns}).to_csv(selected_features_csv_path, index=False)

        leakage_audit_path = project_path("reports/metrics/xgboost_pruned_top50_leakage_audit.json")
        with leakage_audit_path.open("w", encoding="utf-8") as f:
            json.dump(pruned_audit, f, indent=2)

        importance_df = pd.DataFrame({"feature": feature_columns, "importance": final_model.feature_importances_})
        importance_path = project_path("reports/metrics/xgboost_pruned_top50_feature_importance.csv")
        importance_df.to_csv(importance_path, index=False)

        model_path = project_path("reports/models/xgboost_pruned_top50_model.json")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        final_model.save_model(model_path)

        recursive_debug_sample_path = project_path("reports/metrics/xgboost_pruned_top50_recursive_feature_state_sample.csv")
        test_feature_state_df.head(horizon_hours).to_csv(recursive_debug_sample_path, index=False)

        fig_paths = []
        fig_paths += save_plotly_figure(
            actual_vs_predicted(
                pred_df[pred_df["split"] == "validation"], "XGBoost Pruned Top-50 Validation: Actual vs Predicted"
            ),
            "reports/figures/xgboost_pruned_top50_validation_actual_vs_predicted",
        ).values()
        fig_paths += save_plotly_figure(
            actual_vs_predicted(pred_df[pred_df["split"] == "test"], "XGBoost Pruned Top-50 Test: Actual vs Predicted"),
            "reports/figures/xgboost_pruned_top50_test_actual_vs_predicted",
        ).values()
        fig_paths += save_plotly_figure(
            residual_plot(pred_df, "XGBoost Pruned Top-50 Residuals"),
            "reports/figures/xgboost_pruned_top50_residuals",
        ).values()
        fig_paths += save_plotly_figure(
            horizon_metric_plot(horizon_metrics_df, "rmse", "XGBoost Pruned Top-50 RMSE by Forecast Horizon"),
            "reports/figures/xgboost_pruned_top50_rmse_by_horizon",
        ).values()
        fig_paths += save_plotly_figure(
            feature_importance_plot(importance_df, "XGBoost Pruned Top-50 Feature Importance"),
            "reports/figures/xgboost_pruned_top50_feature_importance",
        ).values()

        shap_artifact_dir = None
        if RUN_SHAP:
            shap_artifacts = generate_xgboost_shap_artifacts(
                model=final_model,
                feature_state_df=test_feature_state_df,
                feature_columns=feature_columns,
                output_dir="reports/shap/xgboost_pruned_top50",
                sample_size=SHAP_SAMPLE_SIZE,
                random_state=random_state,
            )
            shap_artifact_dir = shap_artifacts["shap_dir"]
            mlflow.log_param("shap_actual_sample_size", shap_artifacts["sample_size"])
            mlflow.log_param("shap_n_features_explained", len(feature_columns))
            mlflow.log_param("shap_dependence_plot_count", shap_artifacts["n_dependence_plots"])

        pruned_row = {
            "model_name": MODEL_NAME,
            "top_n_features": TOP_N_FEATURES,
            "validation_MAE": validation_metrics["validation_mae"],
            "validation_RMSE": validation_metrics["validation_rmse"],
            "validation_MAPE": validation_metrics["validation_mape"],
            "test_MAE": test_metrics["test_mae"],
            "test_RMSE": test_metrics["test_rmse"],
            "test_MAPE": test_metrics["test_mape"],
            "peak_error_metric": "",
            "best_params_path": str(best_params_path),
            "best_params_summary": json.dumps(study.best_params),
            "feature_list_path": str(features_path),
            "mlflow_run_id": run.info.run_id,
        }
        comparison_path, summary_path = save_comparison_and_summary(pruned_row, maybe_full_metrics_row())

        artifact_paths = [
            pred_path,
            horizon_metrics_path,
            trials_path,
            best_params_path,
            features_path,
            selected_features_csv_path,
            leakage_audit_path,
            importance_path,
            model_path,
            recursive_debug_sample_path,
            comparison_path,
            summary_path,
            *fig_paths,
        ]
        log_artifacts(artifact_paths)
        if shap_artifact_dir is not None:
            mlflow.log_artifacts(str(shap_artifact_dir), artifact_path="shap_pruned")

        print(f"MLflow run_id: {run.info.run_id}")
        print(f"Model: {MODEL_NAME}")
        print(f"Selected features: {len(feature_columns)}")
        print({**validation_metrics, **test_metrics})


if __name__ == "__main__":
    main()
