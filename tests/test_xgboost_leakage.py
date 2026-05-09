from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from soco_forecasting.config import load_config
from soco_forecasting.data import create_time_splits, get_feature_columns, load_modeling_data
from soco_forecasting.leakage import audit_xgboost_feature_columns
from soco_forecasting.recursive import recursive_backtest_48h_windows_with_features


def test_xgboost_feature_exclusions_and_time_splits():
    config = load_config()
    df = load_modeling_data(config)
    splits = create_time_splits(df, config)
    feature_columns = get_feature_columns(df, config)
    audit = audit_xgboost_feature_columns(df, feature_columns, config)

    assert audit["passed"]
    assert config["target_column"] not in feature_columns
    assert config["datetime_column"] not in feature_columns
    assert config["local_datetime_column"] not in feature_columns
    assert splits.train[config["datetime_column"]].max() < splits.validation[config["datetime_column"]].min()
    assert splits.validation[config["datetime_column"]].max() < splits.test[config["datetime_column"]].min()


def test_demand_lag_features_are_past_only():
    config = load_config()
    target = config["target_column"]
    dt_col = config["datetime_column"]
    df = load_modeling_data(config).set_index(dt_col)
    sample_positions = [500, 5000, len(df) // 2, len(df) - 500]

    for lag_hours in [1, 24, 48, 168]:
        feature = f"{target}_lag_{lag_hours}h"
        for position in sample_positions:
            timestamp = df.index[position]
            expected = df.loc[timestamp - pd.Timedelta(hours=lag_hours), target]
            assert np.isclose(df.iloc[position][feature], expected)


def test_demand_rolling_features_are_prior_timestamp_only():
    config = load_config()
    target = config["target_column"]
    dt_col = config["datetime_column"]
    df = load_modeling_data(config).set_index(dt_col)
    sample_positions = [500, 5000, len(df) // 2, len(df) - 500]

    for window_hours in [3, 6, 24, 168]:
        feature = f"{target}_rolling_mean_{window_hours}h"
        for position in sample_positions:
            timestamp = df.index[position]
            prior_values = df.loc[df.index < timestamp, target].tail(window_hours)
            assert len(prior_values) == window_hours
            assert np.isclose(df.iloc[position][feature], prior_values.mean())

    timestamp = df.index[5000]
    prior_values = df.loc[df.index < timestamp, target].tail(24)
    assert np.isclose(df.loc[timestamp, f"{target}_rolling_std_24h"], prior_values.std())
    assert np.isclose(df.loc[timestamp, f"{target}_rolling_max_24h"], prior_values.max())
    assert np.isclose(df.loc[timestamp, f"{target}_rolling_min_24h"], prior_values.min())


def test_recursive_window_uses_prior_predictions_inside_window_and_actuals_after_window():
    target = "demand_imputed_pudl_mwh"
    dt_col = "datetime_utc"
    feature = f"{target}_lag_1h"
    timestamps = pd.date_range("2025-01-01 00:00:00+00:00", periods=106, freq="h")
    history = pd.DataFrame({dt_col: timestamps[:10], target: np.arange(100.0, 110.0), feature: 0.0})
    forecast = pd.DataFrame({dt_col: timestamps[10:], target: np.arange(1000.0, 1096.0), feature: 0.0})

    class LagPlusTenModel:
        def predict(self, x):
            return x[feature].to_numpy(dtype=float) + 10.0

    pred_df, feature_state_df = recursive_backtest_48h_windows_with_features(
        model=LagPlusTenModel(),
        history_df=history,
        forecast_df=forecast,
        feature_columns=[feature],
        target_col=target,
        datetime_col=dt_col,
        horizon_hours=48,
        model_name="test",
        split_name="validation",
    )

    assert feature_state_df.loc[0, feature] == 109.0
    assert pred_df.loc[0, "predicted"] == 119.0
    assert feature_state_df.loc[1, feature] == 119.0
    assert feature_state_df.loc[1, feature] != forecast.loc[0, target]

    first_second_window_row = feature_state_df[feature_state_df["forecast_horizon_hour"] == 1].iloc[1]
    assert first_second_window_row[feature] == forecast.iloc[47][target]
