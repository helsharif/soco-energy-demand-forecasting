from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(y_true, y_pred, prefix: str = "") -> dict[str, float]:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    error = actual - predicted
    nonzero = actual != 0
    mape = np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100 if np.any(nonzero) else np.nan

    metrics = {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape": float(mape),
    }
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def prediction_frame(datetime, actual, predicted, model_name: str, split_name: str) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "datetime_utc": datetime,
            "actual": actual,
            "predicted": predicted,
            "model": model_name,
            "split": split_name,
        }
    )
    df["residual"] = df["actual"] - df["predicted"]
    df["absolute_error"] = df["residual"].abs()
    return df


def metrics_by_forecast_horizon(predictions: pd.DataFrame) -> pd.DataFrame:
    if "forecast_horizon_hour" not in predictions.columns:
        raise ValueError("predictions must include forecast_horizon_hour.")

    rows = []
    group_cols = ["model", "split", "forecast_horizon_hour"]
    for (model, split, horizon_hour), part in predictions.groupby(group_cols):
        row = {
            "model": model,
            "split": split,
            "forecast_horizon_hour": int(horizon_hour),
            "n_rows": int(len(part)),
        }
        row.update(regression_metrics(part["actual"], part["predicted"]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["model", "split", "forecast_horizon_hour"])
