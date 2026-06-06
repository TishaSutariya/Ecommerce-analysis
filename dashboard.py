"""
Simple Funnel Dashboard - Streamlit Version
NO HTML - Just Basic Python!
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# Page setup
st.set_page_config(
    page_title="Funnel Dashboard",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 E-commerce Funnel Analysis Dashboard")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📊 Choose Page")
    page = st.radio(
        "Select:",
        ["📈 Overview", "🎯 Prediction", "🤖 ML Model", "💡 Insights"]
    )

# ============ CREATE DATA ============
@st.cache_data
def create_data():
    np.random.seed(42)
    n_users = 5000
    
    data = {
        'user_id': range(1, n_users + 1),
        'sessions': np.random.choice([1, 2, 3, 4, 5], size=n_users, 
                                     p=[0.5, 0.25, 0.15, 0.07, 0.03]),
        'days_active': np.random.randint(1, 365, size=n_users),
        'device': np.random.choice([0, 1, 2], size=n_users, p=[0.55, 0.35, 0.10]),
        'traffic_source': np.random.choice([0, 1, 2, 3], size=n_users, 
                                           p=[0.40, 0.30, 0.20, 0.10]),
        'max_step': np.random.choice([1, 2, 3, 4, 5, 6, 7], size=n_users,
                                     p=[0.05, 0.15, 0.30, 0.20, 0.15, 0.10, 0.05]),
        'avg_session_time': np.random.randint(30, 600, size=n_users),
    }
    
    df = pd.DataFrame(data)
    df['converted'] = 0
    df.loc[df['max_step'] == 7, 'converted'] = 1
    df.loc[(df['max_step'] == 6) & (np.random.rand(n_users) < 0.7), 'converted'] = 1
    
    return df

df = create_data()

# Add names
device_names = {0: 'Mobile', 1: 'Desktop', 2: 'Tablet'}
source_names = {0: 'Organic', 1: 'Paid Ad', 2: 'Referral', 3: 'Social'}
step_names = {
    1: 'Opened App', 2: 'Browsed', 3: 'Saw Product',
    4: 'Added to Cart', 5: 'Reached Checkout',
    6: 'Entered Payment', 7: 'Purchased'
}

df['device_name'] = df['device'].map(device_names)
df['source_name'] = df['traffic_source'].map(source_names)

# ============ PAGE 1: OVERVIEW ============
if page == "📈 Overview":
    st.header("📊 Funnel Overview")
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Users", f"{len(df):,}")
    with col2:
        st.metric("Total Buyers", f"{df['converted'].sum():,}")
    with col3:
        st.metric("Conversion Rate", f"{df['converted'].mean()*100:.2f}%")
    with col4:
        st.metric("Avg Session Time", f"{df['avg_session_time'].mean():.0f}s")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Funnel")
        funnel_data = []
        for step in range(1, 8):
            users = (df['max_step'] >= step).sum()
            funnel_data.append({'Stage': step_names[step], 'Users': users})
        funnel_df = pd.DataFrame(funnel_data)
        
        fig = px.bar(funnel_df, x='Users', y='Stage', orientation='h',
                    color='Users', color_continuous_scale='Blues')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.subheader("📱 By Device")
        device_conv = df.groupby('device_name')['converted'].mean() * 100
        
        fig = px.bar(device_conv.reset_index(), x='device_name', y='converted',
                    color='device_name', color_discrete_sequence=['#5B8FC9', '#3B6EA5', '#1F4E79'])
        st.plotly_chart(fig, width='stretch')
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📣 By Traffic")
        source_conv = df.groupby('source_name')['converted'].mean() * 100
        
        fig = px.bar(source_conv.reset_index(), x='source_name', y='converted',
                    color='source_name', color_discrete_sequence=['#8FB3D9', '#5B8FC9', '#3B6EA5', '#1F4E79'])
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.subheader("🥧 Traffic Share")
        traffic = df['source_name'].value_counts().reset_index()
        traffic.columns = ['Source', 'Count']
        
        fig = px.pie(traffic, names='Source', values='Count',
                    color='Source', color_discrete_sequence=['#8FB3D9', '#5B8FC9', '#3B6EA5', '#1F4E79'])
        st.plotly_chart(fig, width='stretch')


# ============ PAGE 2: PREDICTION ============
elif page == "🎯 Prediction":
    st.header("🎯 Predict if User Will Buy")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Enter User Info")
        sessions = st.slider("Sessions", 1, 5, 3)
        days = st.slider("Days Active", 1, 365, 30)
        device = st.selectbox("Device", ['Mobile', 'Desktop', 'Tablet'])
        traffic = st.selectbox("Traffic", ['Organic', 'Paid Ad', 'Referral', 'Social'])
        step = st.select_slider("Max Step", list(step_names.values()), 'Added to Cart')
        time = st.slider("Session Time (s)", 30, 600, 300)
    
    with col2:
        st.subheader("Result")
        
        # Convert to numbers
        dev_num = {'Mobile': 0, 'Desktop': 1, 'Tablet': 2}[device]
        traf_num = {'Organic': 0, 'Paid Ad': 1, 'Referral': 2, 'Social': 3}[traffic]
        step_num = {v: k for k, v in step_names.items()}[step]
        
        # Train model
        features = ['sessions', 'days_active', 'device', 'traffic_source', 
                    'max_step', 'avg_session_time']
        X = df[features]
        y = df['converted']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        
        # Predict
        new_user = pd.DataFrame([{
            'sessions': sessions, 'days_active': days, 'device': dev_num,
            'traffic_source': traf_num, 'max_step': step_num, 'avg_session_time': time
        }])
        
        prob = model.predict_proba(new_user)[0][1] * 100
        
        st.metric("Buy Probability", f"{prob:.1f}%")
        
        if prob > 50:
            st.success("✅ LIKELY TO BUY!")
            st.info("→ Send free shipping offer")
        elif prob > 25:
            st.warning("⚠️ MIGHT BUY")
            st.info("→ Send 10% discount")
        else:
            st.error("❌ UNLIKELY TO BUY")
            st.info("→ Show onboarding tips")
        
        acc = accuracy_score(y_test, model.predict(X_test)) * 100
        st.info(f"Model Accuracy: {acc:.1f}%")


# ============ PAGE 3: ML MODEL ============
elif page == "🤖 ML Model":
    st.header("🤖 Machine Learning Model")
    
    features = ['sessions', 'days_active', 'device', 'traffic_source', 
                'max_step', 'avg_session_time']
    X = df[features]
    y = df['converted']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred) * 100
    roc = roc_auc_score(y_test, y_proba)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc:.1f}%")
    col2.metric("ROC-AUC", f"{roc:.3f}")
    col3.metric("Training", f"{len(X_train):,}")
    col4.metric("Testing", f"{len(X_test):,}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📌 Feature Importance")
        importance = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(importance, x='Importance', y='Feature', orientation='h',
                    color='Importance', color_continuous_scale='Blues')
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.subheader("📊 Confusion Matrix")
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        
        fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                       x=['No Buy', 'Buy'], y=['No Buy', 'Buy'],
                       labels={'x': 'Predicted', 'y': 'Actual'})
        st.plotly_chart(fig, width='stretch')


# ============ PAGE 4: INSIGHTS ============
elif page == "💡 Insights":
    st.header("💡 Business Insights")
    
    device_conv = df.groupby('device_name')['converted'].mean() * 100
    source_conv = df.groupby('source_name')['converted'].mean() * 100
    
    # Biggest drop
    biggest_drop = 0
    biggest_step = 0
    for step in range(2, 7):
        curr = (df['max_step'] >= step).sum()
        prev = (df['max_step'] >= step - 1).sum()
        drop = (prev - curr) / prev * 100
        if drop > biggest_drop:
            biggest_drop = drop
            biggest_step = step
    
    st.subheader("1. Biggest Drop-off")
    st.write(f"**Step {biggest_step}: {step_names[biggest_step]}**")
    st.write(f"{biggest_drop:.1f}% users leave here")
    st.warning("💡 Fix: Make this step easier")
    
    st.markdown("---")
    
    st.subheader("2. Best Device")
    st.write(f"🏆 **{device_conv.idxmax()}** - {device_conv.max():.1f}%")
    st.write(f"❌ Worst: {device_conv.idxmin()} - {device_conv.min():.1f}%")
    st.success("💡 Fix: Improve worst device")
    
    st.markdown("---")
    
    st.subheader("3. Best Traffic")
    st.write(f"🏆 **{source_conv.idxmax()}** - {source_conv.max():.1f}%")
    st.success("💡 Invest more here!")
    
    st.markdown("---")
    
    st.subheader("4. Model Score")
    features = ['sessions', 'days_active', 'device', 'traffic_source', 
                'max_step', 'avg_session_time']
    X = df[features]
    y = df['converted']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test)) * 100
    
    st.write(f"🤖 **Accuracy: {acc:.1f}%**")
    st.write("✅ Good for predicting buyers!")

# Footer
st.markdown("---")
st.write("\n🚀 Funnel Dashboard | Built with Streamlit")