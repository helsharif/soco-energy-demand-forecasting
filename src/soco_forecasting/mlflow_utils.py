from __future__ import annotations

from pathlib import Path

import mlflow

from .config import project_path


def setup_mlflow(config: dict) -> None:
    mlflow_db_path = project_path("mlflow.db").as_posix()
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")
    mlflow.set_experiment(config["mlflow_experiment_name"])


def log_split_manifest(manifest: dict) -> None:
    for split_name, split_info in manifest["splits"].items():
        mlflow.log_param(f"{split_name}_start", split_info["start"])
        mlflow.log_param(f"{split_name}_end", split_info["end"])
        mlflow.log_param(f"{split_name}_n_rows", split_info["n_rows"])


def log_artifacts(paths: list[str | Path]) -> None:
    for path in paths:
        p = project_path(path)
        if p.exists():
            mlflow.log_artifact(str(p))
