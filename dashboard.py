"""
Simple Funnel Dashboard
Student-Friendly Version
Blue Colors Only
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Funnel Dashboard", page_icon="🚀", layout="wide")

st.title("🚀 Funnel Dashboard")
st.markdown("---")

with st.sidebar:
    page = st.radio("Choose:", ["Overview", "Prediction", "ML Model", "Insights"])

df = pd.read_csv("data/fake_ecommerce_data.csv")

df["converted"] = 0
df.loc[df["max_step"] == 7, "converted"] = 1
df.loc[(df["max_step"] == 6) & (np.random.rand(len(df)) < 0.7), "converted"] = 1

df["device_name"] = df["device"].map({0: "Mobile", 1: "Desktop", 2: "Tablet"})
df["source_name"] = df["traffic_source"].map({0: "Organic", 1: "Paid Ad", 2: "Referral", 3: "Social"})

step_names = {
    1: "Opened App",
    2: "Browsed",
    3: "Saw Product",
    4: "Added to Cart",
    5: "Reached Checkout",
    6: "Entered Payment",
    7: "Purchased"
}

blue_colors = ["#6CB5FA", "#A9CCE3", "#7FB3D5", "#5499C7", "#4CB7FF", "#34ADFE", "#049BFA"]

X = df[["sessions", "days_active", "device", "traffic_source", "max_step", "avg_session_time"]]
y = df["converted"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

if page == "Overview":
    st.header("📊 Funnel Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Users", len(df))
    col2.metric("Total Buyers", int(df["converted"].sum()))
    col3.metric("Conversion Rate", f"{df['converted'].mean() * 100:.2f}%")

    st.markdown("---")

    funnel_data = []
    for step in range(1, 8):
        users = (df["max_step"] >= step).sum()
        funnel_data.append({"Stage": step_names[step], "Users": users})

    funnel_df = pd.DataFrame(funnel_data)

    fig1 = px.bar(
        funnel_df,
        x="Users",
        y="Stage",
        orientation="h",
        color="Users",
        color_continuous_scale="Blues"
    )
    fig1.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig1, width="stretch")

    st.subheader("📱 Conversion by Device")
    device_conv = df.groupby("device_name")["converted"].mean() * 100
    fig2 = px.bar(
        device_conv.reset_index(),
        x="device_name",
        y="converted",
        color="device_name",
        color_discrete_sequence=blue_colors[:3]
    )
    st.plotly_chart(fig2, width="stretch")

    st.subheader("📣 Conversion by Traffic Source")
    source_conv = df.groupby("source_name")["converted"].mean() * 100
    fig3 = px.bar(
        source_conv.reset_index(),
        x="source_name",
        y="converted",
        color="source_name",
        color_discrete_sequence=blue_colors[:4]
    )
    st.plotly_chart(fig3, width="stretch")

    st.subheader("🥧 Traffic Distribution")
    traffic_dist = df["source_name"].value_counts().reset_index()
    traffic_dist.columns = ["Source", "Count"]
    fig4 = px.pie(
        traffic_dist,
        names="Source",
        values="Count",
        color_discrete_sequence=blue_colors[:4]
    )
    st.plotly_chart(fig4, width="stretch")

elif page == "Prediction":
    st.header("🎯 Predict if User Will Buy")

    sessions = st.slider("Sessions", 1, 5, 3)
    days_active = st.slider("Days Active", 1, 365, 30)
    device = st.selectbox("Device", ["Mobile", "Desktop", "Tablet"])
    traffic = st.selectbox("Traffic Source", ["Organic", "Paid Ad", "Referral", "Social"])
    step = st.select_slider("Max Step Reached", options=list(step_names.values()), value="Added to Cart")
    avg_time = st.slider("Avg Session Time (seconds)", 30, 600, 300)

    predict = st.button("🔮 Predict", type="primary", use_container_width=True)

    if predict:
        device_num = {"Mobile": 0, "Desktop": 1, "Tablet": 2}[device]
        traffic_num = {"Organic": 0, "Paid Ad": 1, "Referral": 2, "Social": 3}[traffic]
        step_num = {v: k for k, v in step_names.items()}[step]

        new_user = pd.DataFrame([{
            "sessions": sessions,
            "days_active": days_active,
            "device": device_num,
            "traffic_source": traffic_num,
            "max_step": step_num,
            "avg_session_time": avg_time
        }])

        prob = model.predict_proba(new_user)[0][1] * 100
        st.metric("Buy Probability", f"{prob:.1f}%")

        if prob > 50:
            st.success("✅ LIKELY TO BUY")
            st.info("→ Send free shipping offer")
        elif prob > 25:
            st.warning("⚠️ MIGHT BUY")
            st.info("→ Send 10% discount")
        else:
            st.error("❌ UNLIKELY TO BUY")
            st.info("→ Show onboarding tips")
    else:
        st.info("Click Predict to see the result.")

elif page == "ML Model":
    st.header("🤖 ML Model")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100

    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{acc:.1f}%")
    col2.metric("Training Samples", len(X_train))

    st.markdown("---")

    importance = pd.DataFrame({
        "Feature": ["sessions", "days_active", "device", "traffic_source", "max_step", "avg_session_time"],
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    fig5 = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig5, width="stretch")

elif page == "Insights":
    st.header("💡 Business Insights")

    device_conv = df.groupby("device_name")["converted"].mean() * 100
    source_conv = df.groupby("source_name")["converted"].mean() * 100

    biggest_drop = 0
    biggest_step = 0
    for step in range(2, 7):
        curr = (df["max_step"] >= step).sum()
        prev = (df["max_step"] >= step - 1).sum()
        drop = (prev - curr) / prev * 100
        if drop > biggest_drop:
            biggest_drop = drop
            biggest_step = step
    
    st.write(f"1. Biggest drop-off: {step_names[biggest_step]} → {biggest_drop:.1f}% drop") 
    st.write(f"2. Best device: {device_conv.idxmax()} - {device_conv.max():.1f}%")
    st.write(f"3. Best traffic source: {source_conv.idxmax()} - {source_conv.max():.1f}%")
    st.write(f"4. Model accuracy: {accuracy_score(y_test, model.predict(X_test)) * 100:.1f}%")

st.markdown("---")
st.write("🚀 Funnel Dashboard | Built with Streamlit")