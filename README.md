# Ecommerce Funnel Analysis

A Streamlit-based data science project for analyzing user funnels, identifying drop-off points, predicting conversion probability, and generating business insights for ecommerce growth.

## Features
- Funnel visualization and drop-off analysis.
- ML-based conversion probability prediction.
- Device, traffic source, and regional performance views.
- Business recommendations with ROI-focused insights.
- Dark-mode-friendly UI with a professional blue theme.

## Requirements
Install the following Python dependencies from `requirements.txt`:

- pandas>=1.5.0
- numpy>=1.23.0
- scipy>=1.9.0
- scikit-learn>=1.2.0
- xgboost>=1.7.0
- lightgbm>=3.3.0
- matplotlib>=3.6.0
- seaborn>=0.12.0
- plotly>=5.13.0
- streamlit>=1.25.0
- streamlit-plotly-events>=0.0.6
- sqlalchemy>=2.0.0
- psycopg2-binary>=2.9.0
- pymysql>=1.0.0
- python-dateutil>=2.8.0
- pytz>=2023.3
- requests>=2.31.0
- Flask>=2.3.0
- flask-cors>=4.0.0
- joblib>=1.3.0
- python-dotenv>=1.0.0
- gunicorn>=21.0.0
- pytest>=7.0.0
- black>=23.0.0
- pylint>=2.16.0
- jupyter>=1.0.0
- optuna>=3.0.0
- shap>=0.41.0
- imbalanced-learn>=0.11.0

## Project Structure
- `streamlit/app.py` — main Streamlit dashboard.
- `streamlit/config.toml` — Streamlit theme settings.
- `data/` — raw and processed datasets.
- `sql/` — SQL analysis scripts.
- `notebooks/` — exploratory analysis notebooks.
- `requirements.txt` — Python dependencies.
- `banner1.png` — dashboard banner image.
- `funnel_analysis.png` — analysis visualization.
- `conversion_heatmap.png` — heatmap visualization.

## Installation
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## How to Run
```bash
streamlit run streamlit/app.py
```

## What’s Included
- Executive dashboard with KPI cards.
- Conversion funnel visualization.
- Conversion prediction interface.
- Analytics by device, region, and traffic source.
- Business insights and recommendations.

## Notes
- This project is designed for portfolio and interview presentation.
- The UI uses a blue color palette for a clean professional look.
- The sidebar is styled to remain readable in dark mode.
