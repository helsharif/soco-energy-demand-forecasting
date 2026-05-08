from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from soco_forecasting.config import ensure_artifact_dirs, load_config
from soco_forecasting.data import (
    create_time_splits,
    get_feature_columns,
    load_modeling_data,
    save_split_manifest,
    save_split_summary_csv,
    validate_leakage_safe_feature_names,
)


def main() -> None:
    config = load_config()
    ensure_artifact_dirs(config)
    df = load_modeling_data(config)
    feature_columns = get_feature_columns(df, config)
    validate_leakage_safe_feature_names(feature_columns)
    splits = create_time_splits(df, config)

    manifest_path = save_split_manifest(splits.manifest, "reports/splits/time_split_manifest.json")
    summary_path = save_split_summary_csv(splits.manifest, "reports/splits/time_split_summary.csv")

    print(f"Saved split manifest: {manifest_path}")
    print(f"Saved split summary: {summary_path}")
    for name, info in splits.manifest["splits"].items():
        print(f"{name}: {info['n_rows']:,} rows | {info['start']} to {info['end']}")


if __name__ == "__main__":
    main()
