# XGBoost Leakage Audit

## Conclusion

Leakage-free based on automated feature-name, split, and recursive-evaluation checks.

## Feature Audit

- Target column: `demand_imputed_pudl_mwh`
- Selected XGBoost features: 100
- Target column selected as feature: False
- Raw datetime columns selected as features: False
- Future-looking selected features: False
- Target duplicate selected features: False
- Suspicious or explicitly excluded source columns: `datetime_utc`, `datetime_local`, `demand_imputed_pudl_mwh`

The selected feature list allows target-derived demand features only when they are explicit past lags or past rolling-window statistics. Columns with names suggesting future information, such as `future`, `lead`, `next`, `target`, `actual`, `t_plus`, `ahead`, or centered rolling windows, are excluded.

## Split Audit

- Train: 2015-07-08 06:00:00+00:00 to 2023-12-31 23:00:00+00:00
- Validation: 2024-01-01 00:00:00+00:00 to 2024-12-31 23:00:00+00:00
- Test: 2025-01-01 00:00:00+00:00 to 2026-02-01 06:00:00+00:00

The split is sequential and time-aware: training occurs before validation, and validation occurs before test. No random train/test split is used.

## Lag And Rolling Feature Audit

Training lag and rolling demand features are historical features. Automated tests verify that sampled lag features equal demand from exactly the stated number of prior hours, and sampled rolling demand features are computed from timestamps strictly before the forecast timestamp.

During recursive validation/test evaluation, target-derived lag and rolling features are recomputed from the evaluator's working history. At the start of each 48-hour window, the history contains observed demand available through the forecast origin. Inside the 48-hour window, earlier forecasted hours are inserted as predictions, not actual future demand.

## Recursive Evaluation Audit

Actual validation/test demand is used for scoring each forecasted timestamp, but it is not inserted into the feature-building history until the full 48-hour forecast window is complete. This prevents later horizons inside the same window from seeing actual future demand.

## Weather And Calendar Features

Calendar features are deterministic and forecastable. Weather features are used according to the project assumption that recorded historical weather in `data/soco_modeling_dataset.csv` acts as a proxy for forecast weather during backtesting. In an operational deployment, these values should be replaced by weather forecasts available at prediction time.

## MLflow Diagnostics

The final XGBoost run logs the feature list, excluded suspicious columns, horizon-level error table and plot, recursive predictions, and a sample of recursive feature states for one 48-hour forecast window.

## Feature Count

Final XGBoost feature count: 100
