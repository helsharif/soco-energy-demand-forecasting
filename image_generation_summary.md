# SOCO 48-Hour Energy Demand Forecasting

## Project Summary

SOCO 48-Hour Energy Demand Forecasting is a short-term electricity demand forecasting project focused on predicting hourly load for the Southern Company (SOCO) balancing authority over the next 48 hours. The workflow combines historical electricity demand patterns, regional weather data, calendar effects, lag features, rolling-window statistics, and statistical and machine learning forecasting models to evaluate how well different approaches capture grid demand behavior.

## Core Technical Workflow

- Data ingestion from historical SOCO hourly demand records.
- Regional weather integration, including temperature, humidity, degree-hour, and seasonal signals.
- Time-series feature engineering with lag variables, rolling-window statistics, calendar encodings, and weather-derived features.
- Time-aware train/validation/test splitting to avoid leakage and preserve chronological order.
- Model training and hyperparameter tuning for statistical and machine learning forecasting models.
- MLflow experiment tracking for parameters, metrics, artifacts, model outputs, and diagnostics.
- Model evaluation using MAE, RMSE, MAPE, residual analysis, and 48-hour horizon error diagnostics.
- Interactive Streamlit dashboard for portfolio-ready model comparison and result exploration.

## Models Compared

- **SARIMAX:** Statistical demand-only baseline using historical load structure and seasonality.
- **Prophet:** Interpretable time-series forecasting model with trend, seasonality, holidays, and targeted weather regressors.
- **XGBoost Full:** Machine learning model using the full engineered feature set, including demand lags, rolling statistics, calendar features, weather variables, CDH/HDH, and humidity signals.
- **XGBoost Pruned Top-50:** Optimized feature-pruned XGBoost model using the top 50 selected features to test whether a simpler model can preserve performance.

## Key Visual Concepts

- Electric grid infrastructure, transmission lines, or a regional power network.
- A modern city skyline or service territory at dusk, implying electricity demand and grid operations.
- A clear 48-hour forecast horizon, represented visually as a future-looking time window.
- Actual vs predicted demand time-series curves with multiple model traces.
- Weather signals such as temperature, humidity, dew point, seasonal cycles, cooling degree hours, and heating degree hours.
- Model comparison cards for SARIMAX, Prophet, XGBoost Full, and XGBoost Pruned Top-50.
- Forecasting metrics such as RMSE, MAE, and MAPE shown as clean, high-level visual indicators.
- A dashboard, analytics control room, or data science workflow aesthetic that feels practical and deployment-ready.

## Suggested Infographic Layout

- **Top:** Bold project title with a concise subtitle about 48-hour SOCO electricity demand forecasting.
- **Left:** Data sources and feature engineering pipeline, showing load data, regional weather, calendar features, lags, and rolling statistics flowing into the modeling workflow.
- **Center:** Large actual vs predicted demand line chart with a highlighted 48-hour forecast window.
- **Right:** Model comparison panel with four model cards and simple metric indicators for RMSE, MAE, and MAPE.
- **Bottom:** Portfolio workflow strip showing MLflow tracking, leakage-aware evaluation, Streamlit dashboard, and deployment-ready presentation.

## Visual Style Guidance

- Modern data science portfolio aesthetic.
- Clean, high-resolution, professional, vivid, and not cartoonish.
- Use electric blue, orange, teal, purple, black, and dark gray accents.
- Include subtle gridlines, chart overlays, small icons, and energy-sector visual elements.
- Use realistic but stylized power infrastructure and dashboard visuals.
- Make the composition suitable as a wide hero image, GitHub README visual, project thumbnail, or LinkedIn post graphic.
- Prefer visual storytelling over dense labels or small text.

## Final Image-Generation Prompt

Create a wide-format professional infographic hero image for a data science portfolio project titled "SOCO 48-Hour Energy Demand Forecasting." The image should visually summarize a machine learning workflow that forecasts hourly electricity demand for the Southern Company balancing authority over the next 48 hours. Show a modern electric grid and city skyline in the background, with subtle transmission lines and an energy control-room atmosphere. In the center, feature a clean actual-vs-predicted demand line chart with multiple colored forecast traces and a highlighted 48-hour future horizon. On the left, show a simplified data pipeline with icons for historical load data, regional weather, calendar effects, lag features, and rolling-window statistics. On the right, show four clean model comparison cards representing SARIMAX, Prophet, XGBoost Full, and XGBoost Pruned Top-50, with simple metric indicators for RMSE, MAE, and MAPE. Along the bottom, include a compact workflow strip suggesting MLflow experiment tracking, leakage-aware time-series evaluation, and an interactive Streamlit dashboard. Use a modern data science portfolio style with electric blue, orange, teal, purple, black, and dark gray accents. Make it polished, high-resolution, vivid, professional, and suitable for a GitHub README hero image, portfolio website thumbnail, and LinkedIn project post. Emphasize charts, energy infrastructure, forecasting, and dashboard analytics. Keep text minimal and avoid tiny labels; rely on clear visual storytelling.

## Negative Prompt / Avoid List

- Avoid cluttered text or dense paragraphs inside the image.
- Avoid unrealistic sci-fi power grids or fantasy energy beams.
- Avoid fake company logos, utility logos, or branded marks.
- Avoid distorted charts, nonsensical axes, or visually misleading plots.
- Avoid illegible dashboard screens filled with tiny unreadable text.
- Avoid generic AI robot imagery unless subtle and clearly secondary.
- Avoid cartoonish styling, exaggerated neon effects, or overly busy compositions.
- Avoid making the image look like a crypto, cybersecurity, or generic AI product ad.
