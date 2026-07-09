from pathlib import Path
import pickle

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Funnel Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "funnel_cleaned.csv"
MODEL_PATH = BASE_DIR / "models" / "random_forest.pkl"

st.title("📊 Funnel Analytics Dashboard")
st.write("Simple and clean dashboard for student project")

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    if "event_time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["event_time"], errors="coerce")
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        df["timestamp"] = pd.NaT

    if "device" in df.columns and "device_type" not in df.columns:
        df["device_type"] = df["device"]

    if "region" in df.columns and "location" not in df.columns:
        df["location"] = df["region"]

    if "channel" in df.columns and "traffic_source" not in df.columns:
        df["traffic_source"] = df["channel"]

    if "purchase_amount" not in df.columns:
        if "revenue" in df.columns:
            df["purchase_amount"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(0)
        else:
            df["purchase_amount"] = 0.0

    if "bounced" not in df.columns:
        df["bounced"] = 0

    if "funnel_stage" in df.columns:
        df["purchase_flag"] = (df["funnel_stage"] == "Purchase").astype(int)
    else:
        df["purchase_flag"] = 0

    return df

@st.cache_resource
def load_model(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

def show_kpis(df):
    total_users = df["user_id"].nunique() if "user_id" in df.columns else len(df)

    if "funnel_stage" in df.columns and "user_id" in df.columns:
        purchases = df.loc[df["funnel_stage"] == "Purchase", "user_id"].nunique()
    else:
        purchases = df["purchase_flag"].sum()

    conversion_rate = (purchases / total_users * 100) if total_users > 0 else 0
    revenue = df["purchase_amount"].sum() if "purchase_amount" in df.columns else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Users", f"{total_users:,}")
    c2.metric("Purchases", f"{purchases:,}")
    c3.metric("Conversion Rate", f"{conversion_rate:.2f}%")

    c4, c5 = st.columns(2)
    c4.metric("Revenue", f"${revenue:,.2f}")
    c5.metric("Rows", f"{len(df):,}")

df = load_data(DATA_PATH)

if df.empty:
    st.error("No data found. Check your CSV file path.")
    st.stop()

menu = st.sidebar.radio("Go to", ["Overview", "Deep Dive", "Model Metrics"])

device_list = sorted(df["device_type"].dropna().astype(str).unique()) if "device_type" in df.columns else []
source_list = sorted(df["traffic_source"].dropna().astype(str).unique()) if "traffic_source" in df.columns else []
location_list = sorted(df["location"].dropna().astype(str).unique()) if "location" in df.columns else []

st.sidebar.subheader("Filters")

valid_dates = df["timestamp"].dropna()
if not valid_dates.empty:
    min_date = valid_dates.min().date()
    max_date = valid_dates.max().date()
    date_range = st.sidebar.date_input("Date range", [min_date, max_date])
else:
    date_range = None

device_filter = st.sidebar.multiselect("Device", device_list)
source_filter = st.sidebar.multiselect("Traffic Source", source_list)
location_filter = st.sidebar.multiselect("Location", location_list)

filtered = df.copy()

if date_range and len(date_range) == 2:
    filtered = filtered[
        (filtered["timestamp"].dt.date >= date_range[0]) &
        (filtered["timestamp"].dt.date <= date_range[1])
    ]

if device_filter and "device_type" in filtered.columns:
    filtered = filtered[filtered["device_type"].isin(device_filter)]

if source_filter and "traffic_source" in filtered.columns:
    filtered = filtered[filtered["traffic_source"].isin(source_filter)]

if location_filter and "location" in filtered.columns:
    filtered = filtered[filtered["location"].isin(location_filter)]

if menu == "Overview":
    st.subheader("Overview")
    show_kpis(filtered)

    if "funnel_stage" in filtered.columns and "user_id" in filtered.columns:
        stage_order = ["Visit Website", "Signup", "Add to Cart", "Checkout", "Purchase"]
        stage_order = [s for s in stage_order if s in filtered["funnel_stage"].dropna().unique().tolist()]

        funnel_df = (
            filtered[filtered["funnel_stage"].isin(stage_order)]
            .groupby("funnel_stage")["user_id"]
            .nunique()
            .reindex(stage_order)
            .fillna(0)
            .reset_index()
        )
        funnel_df.columns = ["funnel_stage", "users"]

        fig = go.Figure(go.Funnel(y=funnel_df["funnel_stage"], x=funnel_df["users"]))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Funnel data is not available.")

    if "location" in filtered.columns:
        st.subheader("Revenue by Location")
        loc_df = filtered.groupby("location", as_index=False).agg(revenue=("purchase_amount", "sum"))
        loc_df = loc_df.sort_values("revenue", ascending=False).head(10)
        fig = px.bar(loc_df, x="location", y="revenue")
        st.plotly_chart(fig, use_container_width=True)

    if "timestamp" in filtered.columns and filtered["timestamp"].notna().any():
        st.subheader("Events Over Time")
        time_df = filtered.copy()
        time_df["date"] = time_df["timestamp"].dt.date
        daily = time_df.groupby("date").size().reset_index(name="events")
        fig = px.line(daily, x="date", y="events", markers=True)
        st.plotly_chart(fig, use_container_width=True)

elif menu == "Deep Dive":
    st.subheader("Deep Dive")
    show_kpis(filtered)

    c1, c2 = st.columns(2)

    with c1:
        st.write("Conversion by Device")
        if "device_type" in filtered.columns and "user_id" in filtered.columns:
            dev = filtered.groupby("device_type", as_index=False).agg(
                users=("user_id", "nunique"),
                purchases=("purchase_flag", "sum")
            )
            dev["conversion_rate"] = np.where(dev["users"] > 0, dev["purchases"] / dev["users"] * 100, 0)
            fig = px.bar(dev, x="device_type", y="conversion_rate")
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.write("Conversion by Traffic Source")
        if "traffic_source" in filtered.columns and "user_id" in filtered.columns:
            src = filtered.groupby("traffic_source", as_index=False).agg(
                users=("user_id", "nunique"),
                purchases=("purchase_flag", "sum")
            )
            src["conversion_rate"] = np.where(src["users"] > 0, src["purchases"] / src["users"] * 100, 0)
            fig = px.bar(src, x="traffic_source", y="conversion_rate")
            st.plotly_chart(fig, use_container_width=True)

    if "funnel_stage" in filtered.columns:
        st.subheader("Funnel Stage Distribution")
        stage_df = filtered["funnel_stage"].value_counts().reset_index()
        stage_df.columns = ["stage", "count"]
        fig = px.pie(stage_df, names="stage", values="count")
        st.plotly_chart(fig, use_container_width=True)

elif menu == "Model Metrics":
    st.subheader("Model Metrics")
    st.write("This page shows model details in a simple way.")

    model = load_model(MODEL_PATH)

    if model is None:
        st.warning("Model file not found.")
        st.stop()

    st.success("Model loaded successfully.")

    if isinstance(model, dict) and "metrics" in model:
        m = model["metrics"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{m.get('test_accuracy', 0):.2%}")
        c2.metric("Precision", f"{m.get('test_precision', 0):.2%}")
        c3.metric("Recall", f"{m.get('test_recall', 0):.2%}")
    else:
        st.info("Saved metrics not available in the model file.")