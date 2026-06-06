"""
Simple Funnel Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("🚀 Funnel Dashboard")
st.markdown("---")

# Sidebar
with st.sidebar:
    page = st.radio("Choose:", ["Overview", "Prediction", "ML Model", "Insights"])

# LOAD DATA
df = pd.read_csv('data/fake_ecommerce_data.csv')

# Add converted
df['converted'] = 0
df.loc[df['max_step'] == 7, 'converted'] = 1
df.loc[(df['max_step'] == 6) & (np.random.rand(len(df)) < 0.7), 'converted'] = 1

# Add names
df['device_name'] = df['device'].map({0: 'Mobile', 1: 'Desktop', 2: 'Tablet'})
df['source_name'] = df['traffic_source'].map({0: 'Organic', 1: 'Paid Ad', 2: 'Referral', 3: 'Social'})

step_names = {
    1: 'Opened App', 2: 'Browsed', 3: 'Saw Product',
    4: 'Added to Cart', 5: 'Reached Checkout',
    6: 'Entered Payment', 7: 'Purchased'
}

# Train model
X = df[['sessions', 'days_active', 'device', 'traffic_source', 'max_step', 'avg_session_time']]
y = df['converted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# ============ OVERVIEW ============
if page == "Overview":
    st.header("📊 Funnel Overview")
    
    st.metric("Total Users", len(df))
    st.metric("Total Buyers", df['converted'].sum())
    st.metric("Conversion Rate", f"{df['converted'].mean()*100:.2f}%")
    
    st.markdown("---")
    
    funnel_data = []
    for step in range(1, 8):
        users = (df['max_step'] >= step).sum()
        funnel_data.append({'Stage': step_names[step], 'Users': users})
    
    fig = px.bar(funnel_data, x='Users', y='Stage', orientation='h', color='Users')
    st.plotly_chart(fig, width='stretch')
    
    device_conv = df.groupby('device_name')['converted'].mean() * 100
    fig = px.bar(device_conv.reset_index(), x='device_name', y='converted', color='device_name')
    st.plotly_chart(fig, width='stretch')
    
    source_conv = df.groupby('source_name')['converted'].mean() * 100
    fig = px.bar(source_conv.reset_index(), x='source_name', y='converted', color='source_name')
    st.plotly_chart(fig, width='stretch')

# ============ PREDICTION ============
elif page == "Prediction":
    st.header("🎯 Predict if User Will Buy")
    
    sessions = st.slider("Sessions", 1, 5, 3)
    days = st.slider("Days Active", 1, 365, 30)
    device = st.selectbox("Device", ['Mobile', 'Desktop', 'Tablet'])
    traffic = st.selectbox("Traffic Source", ['Organic', 'Paid Ad', 'Referral', 'Social'])
    step = st.select_slider("Max Step Reached", list(step_names.values()), 'Added to Cart')
    time = st.slider("Avg Session Time (seconds)", 30, 600, 300)
    
    # Predict button
    predict_button = st.button("🔮 Predict", type="primary", use_container_width=True)
    
    if predict_button:
        dev_num = {'Mobile': 0, 'Desktop': 1, 'Tablet': 2}[device]
        traf_num = {'Organic': 0, 'Paid Ad': 1, 'Referral': 2, 'Social': 3}[traffic]
        step_num = {v: k for k, v in step_names.items()}[step]
        
        new_user = [[sessions, days, dev_num, traf_num, step_num, time]]
        prob = model.predict_proba(new_user)[0][1] * 100
        
        st.markdown("---")
        st.metric("Buy Probability", f"{prob:.1f}%")
        
        if prob > 50:
            st.success("✅ LIKELY TO BUY!")
            st.info("→ Send free shipping offer")
        elif prob > 25:
            st.warning("⚠️ MIGHT BUY")
            st.info("→ Send 10% discount code")
        else:
            st.error("❌ UNLIKELY TO BUY")
            st.info("→ Show onboarding tips to engage them")
        
        acc = accuracy_score(y_test, model.predict(X_test)) * 100
        st.info(f"Model Accuracy: {acc:.1f}%")
    else:
        st.info("👆 Click 'Predict' to see the result!")

# ============ ML MODEL ============
elif page == "ML Model":
    st.header("🤖 Machine Learning Model")
    
    acc = accuracy_score(y_test, model.predict(X_test)) * 100
    st.metric("Accuracy", f"{acc:.1f}%")
    
    imp = pd.DataFrame({
        'Feature': ['sessions', 'days_active', 'device', 'traffic_source', 'max_step', 'avg_session_time'],
        'Importance': model.feature_importances_
    })
    fig = px.bar(imp, x='Importance', y='Feature', orientation='h', color='Importance')
    st.plotly_chart(fig, width='stretch')

# ============ INSIGHTS ============
elif page == "Insights":
    st.header("💡 Business Insights")
    
    device_conv = df.groupby('device_name')['converted'].mean() * 100
    source_conv = df.groupby('source_name')['converted'].mean() * 100
    
    st.write("1. 🏆 Best Device:", device_conv.idxmax(), f"- {device_conv.max():.1f}% conversion")
    st.write("2. ❌ Worst Device:", device_conv.idxmin(), f"- {device_conv.min():.1f}% conversion")
    st.write("3. 🏆 Best Traffic:", source_conv.idxmax(), f"- {source_conv.max():.1f}% conversion")
    acc = accuracy_score(y_test, model.predict(X_test)) * 100
    st.write("4. 🤖 Model Accuracy:", f"{acc:.1f}%")

st.markdown("---")
st.write("🚀 Funnel Dashboard | Built with Streamlit")