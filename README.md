# E-commerce Funnel Analysis Dashboard

A simple student-friendly project built with Python and Streamlit to analyze an e-commerce funnel, predict user conversion, and show business insights.

## Project Overview

This project uses fake e-commerce data to show how users move through a funnel, where they drop off, and how likely they are to buy. It includes a Streamlit dashboard with charts, a prediction page, and machine learning model results.

## Features

- Funnel analysis with blue-colored charts.
- Conversion prediction using Random Forest.
- Model accuracy display.
- Feature importance chart.
- Business insights page.
- Simple beginner-friendly code.

## Files

- `01_funnel_analysis.py` - Main analysis script.
- `dashboard.py` - Streamlit dashboard app.
- `data/fake_ecommerce_data.csv` - Generated dataset.
- `funnel_charts.png` - Saved funnel charts.
- `confusion_matrix.png` - Saved confusion matrix chart.
- `requirements.txt` - Python package list.

## Requirements

Install the required Python packages using:

```powershell
pip install -r requirements.txt
```
## Dashboard Pages

### Overview
Shows funnel charts, conversion by device, conversion by traffic source, and traffic distribution.

### Prediction
Lets the user enter customer details and predict buy probability.

### ML Model
Shows model accuracy and feature importance.

### Insights
Shows the biggest funnel drop-off, best device, best traffic source, and model performance.

## Notes

- The data used in this project is fake and generated for learning.
- The dashboard reads from the saved CSV file.
- The charts use only blue shades for a clean look.

## Future Improvements

- Add real e-commerce data.
- Improve the model with more features.
- Add login and customer segmentation.
- Export dashboard results to CSV or PDF.

## License
This project is for educational use.
