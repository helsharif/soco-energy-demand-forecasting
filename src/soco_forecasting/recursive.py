from __future__ import annotations

import re

import pandas as pd


def target_derived_feature_columns(feature_columns: list[str], target_col: str) -> tuple[list[str], list[str]]:
    lag_pattern = re.compile(rf"^{re.escape(target_col)}_lag_(\d+)h$")
    rolling_pattern = re.compile(rf"^{re.escape(target_col)}_rolling_(mean|max|min|std)_(\d+)h$")
    lag_cols = [col for col in feature_columns if lag_pattern.match(col)]
    rolling_cols = [col for col in feature_columns if rolling_pattern.match(col)]
    return lag_cols, rolling_cols


def _compute_target_feature(
    feature_name: str,
    timestamp: pd.Timestamp,
    history: pd.Series,
    target_col: str,
) -> float:
    lag_match = re.match(rf"^{re.escape(target_col)}_lag_(\d+)h$", feature_name)
    if lag_match:
        lag_hours = int(lag_match.group(1))
        lag_timestamp = timestamp - pd.Timedelta(hours=lag_hours)
        if lag_timestamp not in history.index:
            raise ValueError(f"Missing history for {feature_name} at {lag_timestamp}.")
        return float(history.loc[lag_timestamp])

    rolling_match = re.match(rf"^{re.escape(target_col)}_rolling_(mean|max|min|std)_(\d+)h$", feature_name)
    if rolling_match:
        statistic = rolling_match.group(1)
        window_hours = int(rolling_match.group(2))
        prior_values = history.loc[history.index < timestamp].tail(window_hours)
        if len(prior_values) < window_hours:
            raise ValueError(
                f"Need {window_hours} prior observations for {feature_name} at {timestamp}; "
                f"found {len(prior_values)}."
            )
        if statistic == "mean":
            return float(prior_values.mean())
        if statistic == "max":
            return float(prior_values.max())
        if statistic == "min":
            return float(prior_values.min())
        if statistic == "std":
            return float(prior_values.std())

    raise ValueError(f"Unsupported target-derived feature: {feature_name}")


def make_recursive_feature_row(
    row: pd.Series,
    timestamp: pd.Timestamp,
    history: pd.Series,
    feature_columns: list[str],
    target_col: str,
) -> pd.DataFrame:
    feature_values = row[feature_columns].copy()
    lag_cols, rolling_cols = target_derived_feature_columns(feature_columns, target_col)
    for feature_name in [*lag_cols, *rolling_cols]:
        feature_values.loc[feature_name] = _compute_target_feature(feature_name, timestamp, history, target_col)
    feature_values = pd.to_numeric(feature_values)
    return pd.DataFrame([feature_values], columns=feature_columns)


def recursive_backtest_48h_windows(
    model,
    history_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    feature_columns: list[str],
    target_col: str,
    datetime_col: str,
    horizon_hours: int,
    model_name: str,
    split_name: str,
) -> pd.DataFrame:
    """Forecast a split as repeated recursive 48-hour windows.

    Each window starts with actual demand history through the forecast origin.
    Inside the window, target-derived lag and rolling features are updated with
    prior predictions. At the next origin, the evaluator advances using actual
    observed demand from the completed window.
    """

    history_base = history_df.sort_values(datetime_col).set_index(datetime_col)[target_col].copy()
    forecast_sorted = forecast_df.sort_values(datetime_col).reset_index(drop=True)
    rows: list[dict] = []

    for window_start in range(0, len(forecast_sorted), horizon_hours):
        window = forecast_sorted.iloc[window_start : window_start + horizon_hours].copy()
        origin_timestamp = window[datetime_col].iloc[0] - pd.Timedelta(hours=1)
        working_history = history_base.copy()

        for horizon_step, (_, row) in enumerate(window.iterrows(), start=1):
            timestamp = row[datetime_col]
            x_row = make_recursive_feature_row(row, timestamp, working_history, feature_columns, target_col)
            prediction = float(model.predict(x_row)[0])
            actual = float(row[target_col])
            working_history.loc[timestamp] = prediction

            rows.append(
                {
                    "datetime_utc": timestamp,
                    "actual": actual,
                    "predicted": prediction,
                    "residual": actual - prediction,
                    "absolute_error": abs(actual - prediction),
                    "model": model_name,
                    "split": split_name,
                    "forecast_origin": origin_timestamp,
                    "forecast_horizon_hour": horizon_step,
                }
            )

        actual_window = window.set_index(datetime_col)[target_col]
        history_base = pd.concat([history_base, actual_window]).sort_index()

    return pd.DataFrame(rows)
