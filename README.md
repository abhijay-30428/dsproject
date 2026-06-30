# Traffic Accident Analysis — Data Science Project

## Contents
- `traffic_accidents.csv` — synthetic dataset (8,000 records) modeled on real-world traffic accident data structure
- `Traffic_Accident_Analysis.ipynb` — full Jupyter notebook (EDA, feature engineering, modeling, results) with outputs already executed
- `colab_full_script.py` — single-file version of the same pipeline, ready to paste into Google Colab
- `README.md` — this file

## Pipeline Overview
1. Data loading & cleaning
2. Exploratory Data Analysis (severity distribution, time patterns, weather/road/light breakdowns, correlation heatmap, hotspot heatmap)
3. Feature engineering (encoding, night/rush-hour flags)
4. Modeling — Logistic Regression baseline + Random Forest classifier predicting accident severity (Minor / Serious / Fatal)
5. Model evaluation (accuracy, F1, confusion matrix, feature importance)
6. Key findings & recommendations

## How to Run
**Locally / Jupyter:** open `Traffic_Accident_Analysis.ipynb` and run all cells (requires pandas, numpy, scikit-learn, matplotlib, seaborn).

**Google Colab:** open a new notebook, paste the contents of `colab_full_script.py` into a cell, and run.

## Using Your Own Data
Replace `traffic_accidents.csv` with a real dataset (e.g. Kaggle "US Accidents", UK DfT STATS19 Road Safety data). Keep similar column names, or update the column references in the feature engineering section:
`date, hour, day_of_week, weather, light_condition, road_type, speed_limit, num_vehicles, junction, alcohol_involved, driver_fatigue, casualties, severity`

## Next Steps / Extensions
- Add geospatial hotspot mapping if latitude/longitude is available
- Try gradient boosting models (XGBoost/LightGBM) with hyperparameter tuning
- Time-series forecasting of accident trends
