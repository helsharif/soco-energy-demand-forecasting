from __future__ import annotations

import pandas as pd
from prophet import Prophet


def to_prophet_frame(df: pd.DataFrame, datetime_col: str, target_col: str) -> pd.DataFrame:
    out = df[[datetime_col, target_col]].copy()
    out[datetime_col] = pd.to_datetime(out[datetime_col]).dt.tz_convert(None)
    out = out.rename(columns={datetime_col: "ds", target_col: "y"})
    return out


def build_prophet_model(params: dict, config: dict) -> Prophet:
    return Prophet(
        daily_seasonality=config["prophet"]["daily_seasonality"],
        weekly_seasonality=config["prophet"]["weekly_seasonality"],
        yearly_seasonality=config["prophet"]["yearly_seasonality"],
        **params,
    )


def recursive_prophet_48h_windows(
    history_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    params: dict,
    config: dict,
    target_col: str,
    datetime_col: str,
    horizon_hours: int,
    model_name: str,
    split_name: str,
    max_windows: int | None = None,
) -> pd.DataFrame:
    """Forecast a split as repeated 48-hour Prophet windows.

    Prophet does not expose a cheap state update operation like SARIMAX, so each
    forecast origin refits using the demand history available through that
    origin. After each completed 48-hour window, actual observations are added
    to the history before the next origin.
    """

    history_base = history_df.sort_values(datetime_col)[[datetime_col, target_col]].copy()
    forecast_sorted = forecast_df.sort_values(datetime_col).reset_index(drop=True)
    rows: list[dict] = []

    for window_number, window_start in enumerate(range(0, len(forecast_sorted), horizon_hours), start=1):
        if max_windows is not None and window_number > max_windows:
            break

        window = forecast_sorted.iloc[window_start : window_start + horizon_hours].copy()
        origin_timestamp = window[datetime_col].iloc[0] - pd.Timedelta(hours=1)

        train_p = to_prophet_frame(history_base, datetime_col, target_col)
        future = window[[datetime_col]].copy()
        future[datetime_col] = pd.to_datetime(future[datetime_col]).dt.tz_convert(None)
        future = future.rename(columns={datetime_col: "ds"})

        model = build_prophet_model(params, config)
        model.fit(train_p)
        forecast = model.predict(future)

        for horizon_step, (_, row) in enumerate(window.iterrows(), start=1):
            predicted = float(forecast["yhat"].iloc[horizon_step - 1])
            actual = float(row[target_col])
            rows.append(
                {
                    "datetime_utc": row[datetime_col],
                    "actual": actual,
                    "predicted": predicted,
                    "residual": actual - predicted,
                    "absolute_error": abs(actual - predicted),
                    "model": model_name,
                    "split": split_name,
                    "forecast_origin": origin_timestamp,
                    "forecast_horizon_hour": horizon_step,
                }
            )

        actual_window = window[[datetime_col, target_col]]
        history_base = pd.concat([history_base, actual_window], ignore_index=True).sort_values(datetime_col)

    return pd.DataFrame(rows)
