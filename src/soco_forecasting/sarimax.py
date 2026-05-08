from __future__ import annotations

import pandas as pd


def recursive_sarimax_48h_windows(
    fitted_results,
    forecast_y: pd.Series,
    horizon_hours: int,
    model_name: str,
    split_name: str,
) -> pd.DataFrame:
    """Forecast a target series as repeated 48-hour SARIMAX forecast windows.

    The fitted SARIMAX state is updated with actual observed demand after each
    completed window. This matches the project objective: forecast the next 48
    hourly values from each forecast origin, rather than producing one long
    open-loop forecast across an entire year.
    """

    forecast_y = forecast_y.sort_index()
    current_results = fitted_results
    rows: list[dict] = []

    for window_start in range(0, len(forecast_y), horizon_hours):
        window = forecast_y.iloc[window_start : window_start + horizon_hours]
        origin_timestamp = window.index[0] - pd.Timedelta(hours=1)
        predicted = current_results.forecast(steps=len(window))

        for horizon_step, (timestamp, actual) in enumerate(window.items(), start=1):
            pred_value = float(predicted.loc[timestamp] if timestamp in predicted.index else predicted.iloc[horizon_step - 1])
            actual_value = float(actual)
            rows.append(
                {
                    "datetime_utc": timestamp,
                    "actual": actual_value,
                    "predicted": pred_value,
                    "residual": actual_value - pred_value,
                    "absolute_error": abs(actual_value - pred_value),
                    "model": model_name,
                    "split": split_name,
                    "forecast_origin": origin_timestamp,
                    "forecast_horizon_hour": horizon_step,
                }
            )

        current_results = current_results.extend(window)

    return pd.DataFrame(rows)
