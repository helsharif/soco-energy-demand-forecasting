from __future__ import annotations

from pathlib import Path

import mlflow

from .config import project_path


def setup_mlflow(config: dict) -> None:
    mlflow.set_tracking_uri(project_path("mlruns").as_uri())
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
