from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "modeling_config.json"


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config_path = project_path(path)
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    config["_config_path"] = str(config_path)
    return config


def ensure_artifact_dirs(config: dict[str, Any]) -> None:
    for path in config.get("artifact_dirs", {}).values():
        project_path(path).mkdir(parents=True, exist_ok=True)
