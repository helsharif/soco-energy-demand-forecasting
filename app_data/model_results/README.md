# Dashboard Model Results

This folder contains static, dashboard-ready model results for the Streamlit app.

The deployed app reads from this folder only. It does not connect to MLflow, train models, tune models, or compute SHAP at runtime.

## Contents

- `model_comparison_metrics.csv`: one-row-per-model comparison metrics.
- `manifest.json`: model names and dashboard metadata.
- Per-model folders with:
  - `metadata.json`
  - `metrics.csv`
  - `predictions.csv`
  - `daily_predictions.csv`
  - `horizon_errors.csv`
  - optional `feature_importance.csv`
  - optional `shap_top_features.csv`

## Refresh Workflow

After new MLflow experiments are completed:

1. Identify the best runs for SARIMAX, Prophet, full XGBoost, and pruned XGBoost.
2. Export or copy standardized prediction, metric, horizon-error, feature-importance, and metadata artifacts into this folder.
3. Regenerate `model_comparison_metrics.csv`.
4. Keep app files small enough for GitHub and Streamlit Community Cloud.
5. Commit the updated `app_data/model_results/` files.
6. Redeploy the Streamlit app from GitHub.

For long hourly prediction files, keep full metrics but provide `daily_predictions.csv` for the dashboard default view.
