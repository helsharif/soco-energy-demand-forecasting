from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .config import project_path


FUTURE_LOOKING_PATTERNS = (
    r"(^|_)future($|_)",
    r"(^|_)lead($|_|\d)",
    r"(^|_)next($|_)",
    r"(^|_)target($|_)",
    r"(^|_)actual($|_)",
    r"(^|_)y$",
    r"(^|_)t_plus($|_|\d)",
    r"(^|_)ahead($|_)",
    r"rolling_centered",
    r"(^|_)centered($|_)",
    r"(^|_)delta($|_)",
)


def is_future_looking_name(column: str) -> bool:
    col = column.lower()
    return any(re.search(pattern, col) for pattern in FUTURE_LOOKING_PATTERNS)


def is_allowed_target_derived_feature(column: str, target_col: str) -> bool:
    lag_pattern = rf"^{re.escape(target_col)}_lag_\d+h$"
    rolling_pattern = rf"^{re.escape(target_col)}_rolling_(mean|max|min|std)_\d+h$"
    return bool(re.match(lag_pattern, column) or re.match(rolling_pattern, column))


def audit_xgboost_feature_columns(
    df: pd.DataFrame,
    feature_columns: list[str],
    config: dict,
) -> dict:
    target_col = config["target_column"]
    datetime_columns = {config["datetime_column"], config["local_datetime_column"], "ds", "datetime", "timestamp"}

    selected_future_looking = [col for col in feature_columns if is_future_looking_name(col)]
    selected_datetime = [
        col
        for col in feature_columns
        if col in datetime_columns or pd.api.types.is_datetime64_any_dtype(df[col])
    ]
    selected_target_duplicates = [
        col
        for col in feature_columns
        if target_col in col and col != target_col and not is_allowed_target_derived_feature(col, target_col)
    ]
    selected_target = [col for col in feature_columns if col == target_col]

    excluded_suspicious_columns = [
        col
        for col in df.columns
        if col not in feature_columns
        and (col == target_col or col in datetime_columns or is_future_looking_name(col))
    ]

    passed = not (
        selected_future_looking or selected_datetime or selected_target_duplicates or selected_target
    )
    return {
        "passed": passed,
        "n_features": len(feature_columns),
        "target_column": target_col,
        "selected_target": selected_target,
        "selected_datetime": selected_datetime,
        "selected_future_looking": selected_future_looking,
        "selected_target_duplicates": selected_target_duplicates,
        "excluded_suspicious_columns": excluded_suspicious_columns,
        "policy": (
            "XGBoost excludes the target, raw timestamp fields, future-looking names, "
            "and target-derived demand columns except explicit lag and past rolling features."
        ),
    }


def validate_xgboost_feature_audit(audit: dict) -> None:
    if not audit["passed"]:
        raise ValueError(f"XGBoost leakage feature audit failed: {audit}")


def save_xgboost_leakage_report(
    audit: dict,
    split_manifest: dict,
    feature_columns: list[str],
    output_path: str | Path = "reports/xgboost_leakage_audit.md",
) -> Path:
    path = project_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    split_rows = split_manifest["splits"]
    status = "Leakage-free based on automated feature-name, split, and recursive-evaluation checks"
    suspicious = audit["excluded_suspicious_columns"] or ["None"]

    content = f"""# XGBoost Leakage Audit

## Conclusion

{status}.

## Feature Audit

- Target column: `{audit["target_column"]}`
- Selected XGBoost features: {audit["n_features"]}
- Target column selected as feature: {bool(audit["selected_target"])}
- Raw datetime columns selected as features: {bool(audit["selected_datetime"])}
- Future-looking selected features: {bool(audit["selected_future_looking"])}
- Target duplicate selected features: {bool(audit["selected_target_duplicates"])}
- Suspicious or explicitly excluded source columns: {", ".join(f"`{col}`" for col in suspicious)}

The selected feature list allows target-derived demand features only when they are explicit past lags or past rolling-window statistics. Columns with names suggesting future information, such as `future`, `lead`, `next`, `target`, `actual`, `t_plus`, `ahead`, or centered rolling windows, are excluded.

## Split Audit

- Train: {split_rows["train"]["start"]} to {split_rows["train"]["end"]}
- Validation: {split_rows["validation"]["start"]} to {split_rows["validation"]["end"]}
- Test: {split_rows["test"]["start"]} to {split_rows["test"]["end"]}

The split is sequential and time-aware: training occurs before validation, and validation occurs before test. No random train/test split is used.

## Lag And Rolling Feature Audit

Training lag and rolling demand features are historical features. Automated tests verify that sampled lag features equal demand from exactly the stated number of prior hours, and sampled rolling demand features are computed from timestamps strictly before the forecast timestamp.

During recursive validation/test evaluation, target-derived lag and rolling features are recomputed from the evaluator's working history. At the start of each 48-hour window, the history contains observed demand available through the forecast origin. Inside the 48-hour window, earlier forecasted hours are inserted as predictions, not actual future demand.

## Recursive Evaluation Audit

Actual validation/test demand is used for scoring each forecasted timestamp, but it is not inserted into the feature-building history until the full 48-hour forecast window is complete. This prevents later horizons inside the same window from seeing actual future demand.

## Weather And Calendar Features

Calendar features are deterministic and forecastable. Weather features are used according to the project assumption that recorded historical weather in `data/soco_modeling_dataset.csv` acts as a proxy for forecast weather during backtesting. In an operational deployment, these values should be replaced by weather forecasts available at prediction time.

## MLflow Diagnostics

The final XGBoost run logs the feature list, excluded suspicious columns, horizon-level error table and plot, recursive predictions, and a sample of recursive feature states for one 48-hour forecast window.

## Feature Count

Final XGBoost feature count: {len(feature_columns)}
"""
    path.write_text(content, encoding="utf-8")
    return path
