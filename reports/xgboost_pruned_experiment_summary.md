# XGBoost Pruned Top-50 Experiment

## Purpose

This experiment tests whether a simpler XGBoost model using only the top 50 features from the full XGBoost feature-importance results can preserve most of the recursive 48-hour forecasting performance.

## Feature Selection

Features were selected from the full XGBoost importance artifact, preferring `reports/shap/xgboost/shap_top_features.csv`. Ranked features were filtered through the same leakage audit used by the full XGBoost workflow. Unsafe or unavailable features were skipped.

## Evaluation

The pruned model was tuned independently with Optuna using lightweight direct validation predictions. After hyperparameter selection, it was evaluated once with the same leakage-free recursive 48-hour validation/test procedure used by the full XGBoost model.

## Result

- Validation RMSE: 867.8562804644251
- Test RMSE: 998.9332275679304
- MLflow run id: 89b55254d2c14bf392f539eda28a171a

## Recommendation

The Top-50 pruned model is within about 3% of the full model by test RMSE and is worth considering for promotion.
