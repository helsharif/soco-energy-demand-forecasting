from __future__ import annotations

import pandas as pd
from prophet import Prophet


def get_prophet_regressors(config: dict) -> list[str]:
    return list(config["prophet"].get("regressors", []))


def validate_prophet_regressors(df: pd.DataFrame, config: dict) -> None:
    regressors = get_prophet_regressors(config)
    missing = [col for col in regressors if col not in df.columns]
    if missing:
        raise ValueError(f"Missing Prophet regressor columns: {missing}")
    null_counts = df[regressors].isna().sum() if regressors else pd.Series(dtype=int)
    with_nulls = null_counts[null_counts > 0].to_dict()
    if with_nulls:
        raise ValueError(f"Prophet regressor columns contain null values: {with_nulls}")


def to_prophet_frame(df: pd.DataFrame, datetime_col: str, target_col: str, regressors: list[str]) -> pd.DataFrame:
    out = df[[datetime_col, target_col, *regressors]].copy()
    out[datetime_col] = pd.to_datetime(out[datetime_col]).dt.tz_convert(None)
    out = out.rename(columns={datetime_col: "ds", target_col: "y"})
    return out


def build_prophet_model(params: dict, config: dict) -> Prophet:
    params = params.copy()
    yearly_fourier_order = params.pop("yearly_fourier_order", None)
    yearly_seasonality = config["prophet"]["yearly_seasonality"]
    if yearly_fourier_order is not None and yearly_seasonality:
        yearly_seasonality = int(yearly_fourier_order)

    model = Prophet(
        daily_seasonality=config["prophet"]["daily_seasonality"],
        weekly_seasonality=config["prophet"]["weekly_seasonality"],
        yearly_seasonality=yearly_seasonality,
        **params,
    )
    country_holidays = config["prophet"].get("country_holidays")
    if country_holidays:
        model.add_country_holidays(country_name=country_holidays)
    for regressor in get_prophet_regressors(config):
        model.add_regressor(regressor)
    return model


def direct_prophet_48h_windows(
    train_df: pd.DataFrame,
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
    """Forecast a split directly, then evaluate in 48-hour windows.

    Prophet is a demand-only trend/seasonality model, so it can produce direct
    multi-step forecasts without recursive target-derived features. This helper
    fits Prophet once on the supplied training history, predicts the requested
    timestamps, and labels results in 48-hour windows for horizon diagnostics.
    """

    regressors = get_prophet_regressors(config)
    validate_prophet_regressors(train_df, config)
    validate_prophet_regressors(forecast_df, config)

    train_p = to_prophet_frame(train_df.sort_values(datetime_col), datetime_col, target_col, regressors)
    forecast_sorted = forecast_df.sort_values(datetime_col).reset_index(drop=True)
    if max_windows is not None:
        forecast_sorted = forecast_sorted.iloc[: max_windows * horizon_hours].copy()

    future = forecast_sorted[[datetime_col, *regressors]].copy()
    future[datetime_col] = pd.to_datetime(future[datetime_col]).dt.tz_convert(None)
    future = future.rename(columns={datetime_col: "ds"})

    model = build_prophet_model(params, config)
    model.fit(train_p)
    forecast = model.predict(future)

    rows: list[dict] = []

    for window_start in range(0, len(forecast_sorted), horizon_hours):
        window = forecast_sorted.iloc[window_start : window_start + horizon_hours].copy()
        window_forecast = forecast.iloc[window_start : window_start + len(window)].reset_index(drop=True)
        origin_timestamp = window[datetime_col].iloc[0] - pd.Timedelta(hours=1)

        for horizon_step, (_, row) in enumerate(window.iterrows(), start=1):
            predicted = float(window_forecast["yhat"].iloc[horizon_step - 1])
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

    return pd.DataFrame(rows)
