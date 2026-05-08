from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import project_path


@dataclass(frozen=True)
class SplitData:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    manifest: dict


def load_modeling_data(config: dict) -> pd.DataFrame:
    dataset_path = project_path(config["dataset_path"])
    datetime_col = config["datetime_column"]
    target_col = config["target_column"]

    df = pd.read_csv(dataset_path, parse_dates=[datetime_col, config["local_datetime_column"]])
    df = df.sort_values(datetime_col).reset_index(drop=True)

    if df[datetime_col].isna().any():
        raise ValueError(f"{datetime_col} contains missing timestamps.")
    if df[target_col].isna().any():
        raise ValueError(f"{target_col} contains missing target values.")
    if not df[datetime_col].is_monotonic_increasing:
        raise ValueError(f"{datetime_col} must be strictly time ordered after sorting.")
    if df[datetime_col].duplicated().any():
        raise ValueError(f"{datetime_col} contains duplicate timestamps.")

    return df


def _between(df: pd.DataFrame, datetime_col: str, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    mask = (df[datetime_col] >= start_ts) & (df[datetime_col] <= end_ts)
    return df.loc[mask].copy()


def create_time_splits(df: pd.DataFrame, config: dict) -> SplitData:
    datetime_col = config["datetime_column"]
    split = config["split"]

    train = _between(df, datetime_col, split["train_start"], split["train_end"])
    validation = _between(df, datetime_col, split["validation_start"], split["validation_end"])
    test = _between(df, datetime_col, split["test_start"], split["test_end"])

    if train.empty or validation.empty or test.empty:
        raise ValueError("Train, validation, and test splits must all contain rows.")
    if not (train[datetime_col].max() < validation[datetime_col].min() < validation[datetime_col].max() < test[datetime_col].min()):
        raise ValueError("Splits must be sequential: train before validation before test.")

    manifest = build_split_manifest(config, train, validation, test)
    return SplitData(train=train, validation=validation, test=test, manifest=manifest)


def build_split_manifest(config: dict, train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> dict:
    datetime_col = config["datetime_column"]
    target_col = config["target_column"]

    def describe(name: str, part: pd.DataFrame) -> dict:
        return {
            "name": name,
            "n_rows": int(len(part)),
            "start": str(part[datetime_col].min()),
            "end": str(part[datetime_col].max()),
            "start_index": int(part.index.min()),
            "end_index": int(part.index.max()),
            "target_nulls": int(part[target_col].isna().sum()),
        }

    return {
        "dataset_path": config["dataset_path"],
        "datetime_column": datetime_col,
        "target_column": target_col,
        "forecast_horizon_hours": config["forecast_horizon_hours"],
        "split_policy": "Sequential UTC calendar split; no random sampling.",
        "leakage_policy": (
            "Training uses observations through train_end only; tuning uses validation only; "
            "test is reserved for one final evaluation after model selection."
        ),
        "splits": {
            "train": describe("train", train),
            "validation": describe("validation", validation),
            "test": describe("test", test),
        },
    }


def save_split_manifest(manifest: dict, output_path: str | Path) -> Path:
    path = project_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path


def save_split_summary_csv(manifest: dict, output_path: str | Path) -> Path:
    rows = list(manifest["splits"].values())
    path = project_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def get_feature_columns(df: pd.DataFrame, config: dict) -> list[str]:
    excluded = {
        config["target_column"],
        config["datetime_column"],
        config["local_datetime_column"],
    }
    return [col for col in df.columns if col not in excluded]


def validate_leakage_safe_feature_names(feature_columns: list[str]) -> None:
    suspicious_tokens = ("lead", "future", "next_", "t_plus", "ahead")
    suspicious = [col for col in feature_columns if any(token in col.lower() for token in suspicious_tokens)]
    if suspicious:
        raise ValueError(f"Potential future-looking feature names found: {suspicious}")
