from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import shap
import xgboost as xgb

from .config import project_path
from .plots import save_plotly_figure


def generate_xgboost_shap_artifacts(
    model,
    feature_state_df: pd.DataFrame,
    feature_columns: list[str],
    output_dir: str | Path,
    sample_size: int = 5000,
    random_state: int = 42,
) -> dict[str, Path | int]:
    """Generate SHAP artifacts for recursive XGBoost forecast feature states."""

    shap_dir = project_path(output_dir)
    shap_dir.mkdir(parents=True, exist_ok=True)

    x_all = feature_state_df[feature_columns].copy()
    n_sample = min(sample_size, len(x_all))
    x_sample = x_all.sample(n=n_sample, random_state=random_state)
    x_sample_path = shap_dir / "shap_input_sample.csv"
    x_sample.to_csv(x_sample_path, index=False)

    shap_method = "shap.TreeExplainer"
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_sample)
        shap_values = np.asarray(shap_values)
    except Exception:
        # SHAP can lag newer XGBoost serialization details. XGBoost's
        # pred_contribs output is Tree SHAP; the final column is the bias term.
        shap_method = "xgboost.pred_contribs"
        dmatrix = xgb.DMatrix(x_sample, feature_names=feature_columns)
        contribs = model.get_booster().predict(dmatrix, pred_contribs=True)
        shap_values = np.asarray(contribs[:, :-1])

    shap_values_path = shap_dir / "shap_values_sample.npz"
    np.savez_compressed(shap_values_path, shap_values=shap_values, feature_names=np.asarray(feature_columns))

    mean_abs = np.abs(shap_values).mean(axis=0)
    top_features = (
        pd.DataFrame({"feature": feature_columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    top_features_path = shap_dir / "shap_top_features.csv"
    top_features.to_csv(top_features_path, index=False)

    metadata_path = shap_dir / "shap_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "sample_size": int(n_sample),
                "n_features": int(len(feature_columns)),
                "random_state": int(random_state),
                "shap_method": shap_method,
                "feature_state": "Recursive 48-hour XGBoost forecast features; target-derived lag/rolling features include prior predictions inside each forecast window.",
            },
            f,
            indent=2,
        )

    bar_df = top_features.head(25).sort_values("mean_abs_shap")
    bar_fig = px.bar(
        bar_df,
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        title="XGBoost SHAP Global Feature Importance",
        template="plotly_white",
        labels={"mean_abs_shap": "Mean absolute SHAP value", "feature": "Feature"},
    )
    save_plotly_figure(bar_fig, shap_dir / "shap_global_feature_importance")

    plt.figure()
    shap.summary_plot(shap_values, x_sample, show=False, max_display=30)
    beeswarm_path = shap_dir / "shap_summary_beeswarm.png"
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=180, bbox_inches="tight")
    plt.close()

    dependence_candidates = [
        "CDH_regional",
        "HDH_regional",
        "relative_humidity_2m_regional_mean",
        "demand_imputed_pudl_mwh_lag_24h",
        "demand_imputed_pudl_mwh_lag_48h",
        "demand_imputed_pudl_mwh_lag_168h",
    ]
    dependence_paths = []
    for feature in dependence_candidates:
        if feature not in x_sample.columns:
            continue
        plt.figure()
        shap.dependence_plot(feature, shap_values, x_sample, show=False)
        path = shap_dir / f"shap_dependence_{feature}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=180, bbox_inches="tight")
        plt.close()
        dependence_paths.append(path)

    return {
        "shap_dir": shap_dir,
        "sample_size": int(n_sample),
        "n_features": int(len(feature_columns)),
        "input_sample_path": x_sample_path,
        "values_path": shap_values_path,
        "top_features_path": top_features_path,
        "metadata_path": metadata_path,
        "n_dependence_plots": len(dependence_paths),
    }
