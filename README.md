# 📊 User Journey Funnel Analysis

A Data Science and Machine Learning project that analyzes user behavior across an e-commerce funnel, identifies customer drop-off points, and predicts purchase conversion using a Random Forest model. The project also includes an interactive Streamlit dashboard for visualizing funnel performance and business insights.

---

## 🚀 Project Overview

Understanding how users move through an e-commerce website is essential for improving conversion rates. This project analyzes user interactions from the first website visit to the final purchase by preprocessing raw event data, performing SQL-based business analysis, training a machine learning model, and presenting insights through an interactive dashboard.

The project follows a complete analytics workflow:

* Data preprocessing and cleaning
* Funnel stage mapping
* User behavior analysis
* Machine learning for conversion prediction
* Interactive dashboard visualization
* SQL-based business insights

---

## ✨ Features

* 📁 Cleans and preprocesses raw user journey data
* 🔄 Maps user events into funnel stages
* 📈 Tracks user progression through the sales funnel
* 💥 Detects bounced users
* ⏱ Calculates user session duration
* 🤖 Predicts user conversion using Random Forest
* 📊 Displays key business KPIs
* 📉 Interactive funnel visualization
* 📍 Revenue analysis by location
* 📱 Device-wise conversion analysis
* 🌐 Traffic source analysis
* 🗄 SQL queries for business reporting

---

## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Plotly
* SQL
* Pickle

---

## 📂 Project Structure

```text
User Journey Funnel Analysis/
│
├── dashboard/
│   └── app.py                 # Streamlit Dashboard
│
├── preprocessing/
│   └── load.py                # Data cleaning & preprocessing
│
├── models/
│   ├── train.py               # Random Forest model training
│   └── random_forest.pkl      # Saved trained model
│
├── data/
│   ├── raw/
│   │   └── Funnel_Analysis_Data.csv
│   │
│   └── processed/
│       └── funnel_cleaned.csv
│
├── sql_analysis.sql           # Business SQL queries
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📥 Data Preprocessing

The preprocessing module (`load.py`) performs the following tasks:

* Loads raw user journey data
* Standardizes column names
* Handles missing values
* Removes duplicate records
* Converts event timestamps
* Maps events into funnel stages
* Creates funnel ranking
* Validates user journey sequence
* Calculates session duration
* Detects bounced users
* Creates additional analytical features

The cleaned dataset is stored in:

```text
data/processed/funnel_cleaned.csv
```

---

## 🤖 Machine Learning Model

The project uses a **Random Forest Classifier** to predict whether a user will complete a purchase.

### Input Features

* Device Type
* Traffic Source
* Location
* Session Duration
* Page Views
* Scroll Depth
* Session Count
* Visit Count
* Days Since Last Visit
* Engagement Score
* Bounce Status

### Model Workflow

1. Aggregate user-level data
2. Encode categorical variables
3. Train-Test Split
4. Train Random Forest model
5. Evaluate model performance
6. Save trained model

The trained model is stored as:

```text
models/random_forest.pkl
```

---

## 📊 Streamlit Dashboard

The interactive dashboard includes three sections:

### 1. Overview

* Total Users
* Purchases
* Conversion Rate
* Revenue
* Funnel Chart
* Revenue by Location
* Events Over Time

### 2. Deep Dive

* Device-wise Conversion
* Traffic Source Analysis
* Funnel Stage Distribution
* Interactive Filters

### 3. Model Metrics

Displays machine learning evaluation metrics including:

* Accuracy
* Precision
* Recall

---

## 🗄 SQL Analysis

The project includes SQL queries for business reporting such as:

* Total Users
* Purchase Users
* Device-wise User Distribution
* Revenue by Traffic Source
* Revenue by Product Category

## 📈 Business Insights

This project helps answer business questions such as:

* Where do most users drop off in the funnel?
* Which traffic source generates the highest revenue?
* Which device type has the best conversion rate?
* Which locations generate the highest revenue?
* What factors influence user conversion?

---

## 🎯 Future Improvements

* Deploy the dashboard on Streamlit Community Cloud
* Add more machine learning models for comparison
* Integrate real-time user data
* Build customer segmentation
* Add predictive recommendations
* Improve dashboard with advanced analytics

---
