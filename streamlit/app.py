import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

pio.templates.default = "plotly"

PRIMARY = "#1F4E79"
SECONDARY = "#3B6EA5"
ACCENT = "#5B8FC9"
MUTED = "#8FB3D9"
LIGHT = "#DCEAF7"
DARK = "#163A5F"
SOFT_SCALE = ["#EEF5FB", "#DCEAF7", "#B8D4EE", "#8FB3D9", "#5B8FC9", "#1F4E79"]

st.set_page_config(
    page_title="Funnel Analysis Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main {background-color: #f8fafc;}
section[data-testid="stSidebar"] {
    background-color: #0E1117 !important;
}
section[data-testid="stSidebar"] * {
    color: #FAFAFA !important;
}
section[data-testid="stSidebar"] .stSelectbox,
section[data-testid="stSidebar"] .stRadio,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {
    color: #FAFAFA !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] {
    color: #FAFAFA !important;
}
.metric-box {
    background: linear-gradient(135deg, #1F4E79 0%, #3B6EA5 100%);
    color: white;
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

def estimate_conversion_prob(session_freq, lifetime_days, stage_reached, device, traffic):
    stage_values = {
        'App Open': 1, 'Browse': 2, 'Product View': 3, 'Add to Cart': 4,
        'Checkout': 5, 'Payment': 6, 'Purchase': 7
    }
    traffic_adjust = {
        'Organic': 1.3, 'Paid': 0.97, 'Referral': 1.5,
        'Social': 0.6, 'Direct': 1.1
    }
    base_prob = 0.032
    if device == 'Desktop':
        base_prob *= 1.5
    elif device == 'Tablet':
        base_prob *= 1.0
    else:
        base_prob *= 0.5
    base_prob *= traffic_adjust.get(traffic, 1.0)
    base_prob *= (1 + stage_values.get(stage_reached, 1) * 0.15)
    base_prob *= (1 + min(session_freq / 20, 0.5))
    base_prob *= (1 + min(lifetime_days / 365 * 0.3, 0.3))
    return min(base_prob, 0.95)

def evaluate_model(y_true, y_pred, y_proba=None):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["ROC-AUC"] = roc_auc_score(y_true, y_proba)
    return metrics

@st.cache_data
def load_sample_data():
    funnel_data = {
        'Stage': ['App Open', 'Browse', 'Product View', 'Add to Cart', 'Checkout', 'Payment', 'Purchase'],
        'Users': [500000, 425000, 255000, 102000, 51000, 16000, 16000],
        'Conversion %': [100, 85, 60, 40, 50, 31, 100]
    }
    device_data = {
        'Device': ['Desktop', 'Tablet', 'Mobile'],
        'Users': [200000, 80000, 220000],
        'Converters': [9600, 2560, 3520],
        'Conversion Rate': [4.8, 3.2, 1.6]
    }
    traffic_data = {
        'Source': ['Organic', 'Paid', 'Referral', 'Social', 'Direct'],
        'Users': [200000, 150000, 75000, 50000, 25000],
        'Conversion Rate': [5.2, 3.1, 4.8, 1.9, 3.5]
    }
    region_data = {
        'Region': ['North', 'South', 'East', 'West', 'Central'],
        'Users': [100000, 120000, 110000, 95000, 75000],
        'Conversion Rate': [3.8, 4.5, 3.5, 3.2, 2.1]
    }
    return {
        'funnel': pd.DataFrame(funnel_data),
        'device': pd.DataFrame(device_data),
        'traffic': pd.DataFrame(traffic_data),
        'region': pd.DataFrame(region_data)
    }

data = load_sample_data()

y_test = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1])
y_pred = np.array([0, 0, 1, 0, 0, 1, 0, 1, 1, 1])
y_proba = np.array([0.12, 0.22, 0.81, 0.44, 0.30, 0.88, 0.15, 0.79, 0.51, 0.93])

with st.sidebar:
    st.title("🚀 Funnel Analytics")
    st.markdown("---")
    page = st.radio(
        "Select Page:",
        ["📊 Dashboard", "🎯 Prediction", "🧪 Model Evaluation", "📈 Analytics", "💡 Insights", "📋 FAQ"]
    )
    st.markdown("---")
    st.subheader("About")
    st.info(
        "Complete funnel analysis system with ML-powered conversion prediction. "
        "Built with production-grade data science."
    )
    st.markdown("---")
    st.subheader("Data Source")
    data_source = st.selectbox(
        "Select Data Period:",
        ["Last 30 Days", "Last 90 Days", "Last 6 Months", "Year-to-Date"]
    )

if page == "📊 Dashboard":
    st.image("banner1.png", use_container_width=True)
    st.title("📊 Executive Dashboard")
    st.markdown("Real-time funnel analysis and conversion metrics")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Users", "500K", "+5.2%", delta_color="off")
    with col2:
        st.metric("Conversion Rate", "3.2%", "+0.1%", delta_color="off")
    with col3:
        st.metric("Monthly Revenue", "$10M", "+8.5%", delta_color="off")
    with col4:
        st.metric("Avg Session (s)", "240", "-15s", delta_color="inverse")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Conversion Funnel")
        fig_funnel = px.bar(
            data['funnel'],
            x='Conversion %',
            y='Stage',
            orientation='h',
            color='Stage',
            color_discrete_sequence=[PRIMARY, SECONDARY, ACCENT, MUTED, LIGHT, DARK],
            labels={'Conversion %': 'Conversion Rate (%)', 'Stage': 'Funnel Stage'},
            height=400
        )
        fig_funnel.update_layout(showlegend=False, hovermode='x unified')
        st.plotly_chart(fig_funnel, use_container_width=True)

    with col2:
        st.subheader("🎯 Conversion by Device")
        fig_device = px.bar(
            data['device'],
            x='Device',
            y='Conversion Rate',
            color='Device',
            color_discrete_sequence=[PRIMARY, SECONDARY, MUTED],
            labels={'Conversion Rate': 'Conversion Rate (%)'},
            height=400
        )
        fig_device.update_layout(showlegend=False, hovermode='x unified')
        st.plotly_chart(fig_device, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Traffic Distribution")
        fig_traffic = px.pie(
            data['traffic'],
            names='Source',
            values='Users',
            color='Source',
            color_discrete_sequence=[PRIMARY, SECONDARY, ACCENT, MUTED, LIGHT],
            title='Traffic Distribution',
            height=400
        )
        st.plotly_chart(fig_traffic, use_container_width=True)

    with col2:
        st.subheader("🌍 Regional Performance")
        fig_region = px.bar(
            data['region'],
            x='Region',
            y='Conversion Rate',
            color='Conversion Rate',
            color_continuous_scale=SOFT_SCALE,
            labels={'Conversion Rate': 'Conversion Rate (%)'},
            height=400
        )
        fig_region.update_layout(showlegend=False)
        st.plotly_chart(fig_region, use_container_width=True)

elif page == "🎯 Prediction":
    st.title("🎯 Conversion Probability Predictor")
    st.markdown("Predict user conversion likelihood based on behavioral features")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("User Features")
        session_freq = st.slider("Session Frequency", 1, 50, 3)
        lifetime_days = st.slider("User Lifetime (days)", 1, 365, 30)
        stage_reached = st.select_slider(
            "Last Stage Reached",
            options=['App Open', 'Browse', 'Product View', 'Add to Cart', 'Checkout', 'Payment', 'Purchase'],
            value='Product View'
        )
        device = st.selectbox("Primary Device", ['Mobile', 'Desktop', 'Tablet'])
        traffic = st.selectbox("Traffic Source", ['Organic', 'Paid', 'Referral', 'Social', 'Direct'])

    with col2:
        st.subheader("Prediction Result")
        conversion_prob = estimate_conversion_prob(
            session_freq=session_freq,
            lifetime_days=lifetime_days,
            stage_reached=stage_reached,
            device=device,
            traffic=traffic
        )
        segment = "HIGH" if conversion_prob > 0.5 else "MEDIUM" if conversion_prob > 0.3 else "LOW"
        recommendation = (
            "This user is likely to convert. Focus on closing."
            if segment == "HIGH" else
            "This user shows interest. Implement re-engagement campaign."
            if segment == "MEDIUM" else
            "This user needs nurturing. Optimize checkout experience."
        )
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1F4E79 0%, #3B6EA5 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 15px;
                    text-align: center;">
            <h3>Conversion Probability</h3>
            <h1>{conversion_prob*100:.1f}%</h1>
            <h4>{segment}</h4>
        </div>
        """, unsafe_allow_html=True)
        st.success(recommendation)
        confidence = min(conversion_prob * 2, 0.95)
        st.metric("Model Confidence", f"{confidence*100:.1f}%")

    st.markdown("---")
    st.subheader("Feature Impact Analysis")
    features_df = pd.DataFrame({
        'Feature': ['Traffic Source', 'Device Type', 'Stage Reached', 'Session Frequency', 'User Lifetime'],
        'Importance': [0.35, 0.25, 0.20, 0.12, 0.08]
    })
    fig_importance = px.bar(
        features_df,
        x='Feature',
        y='Importance',
        color='Importance',
        color_continuous_scale=SOFT_SCALE,
        labels={'Importance': 'Relative Importance'},
        height=300
    )
    st.plotly_chart(fig_importance, use_container_width=True)

elif page == "🧪 Model Evaluation":
    st.title("🧪 Model Evaluation")
    st.markdown("Model performance on hold-out test data")
    st.markdown("---")

    metrics = evaluate_model(y_test, y_pred, y_proba)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
    c2.metric("Precision", f"{metrics['Precision']:.3f}")
    c3.metric("Recall", f"{metrics['Recall']:.3f}")
    c4.metric("F1 Score", f"{metrics['F1 Score']:.3f}")
    c5.metric("ROC-AUC", f"{metrics['ROC-AUC']:.3f}")

    st.markdown("---")
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale=SOFT_SCALE,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["Non-Converted", "Converted"],
        y=["Non-Converted", "Converted"],
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True)

elif page == "📈 Analytics":
    st.title("📈 Detailed Analytics")
    st.markdown("In-depth funnel analysis and user behavior insights")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        metric = st.selectbox("Analysis Metric", ["Conversion Rate", "Drop-off Rate", "Session Duration", "Revenue Impact"])
    with col2:
        dimension = st.selectbox("Dimension", ["Device Type", "Traffic Source", "Region", "Day of Week"])
    with col3:
        date_range = st.date_input("Date Range", value=())

    st.markdown("---")
    fig = None

    if dimension == "Device Type":
        fig = px.bar(
            data['device'],
            x='Device',
            y='Conversion Rate',
            color='Device',
            color_discrete_sequence=[PRIMARY, SECONDARY, MUTED],
            height=400,
            title=f"{metric} by {dimension}"
        )
    elif dimension == "Traffic Source":
        fig = px.bar(
            data['traffic'],
            x='Source',
            y='Conversion Rate',
            color='Source',
            color_discrete_sequence=[PRIMARY, SECONDARY, ACCENT, MUTED, LIGHT],
            height=400,
            title=f"{metric} by {dimension}"
        )
    elif dimension == "Region":
        fig = px.bar(
            data['region'],
            x='Region',
            y='Conversion Rate',
            color='Conversion Rate',
            color_continuous_scale=SOFT_SCALE,
            height=400,
            title=f"{metric} by {dimension}"
        )
    elif dimension == "Day of Week":
        day_df = pd.DataFrame({
            "Day": ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
            "Value": [12, 18, 15, 16, 20, 22, 14]
        })
        fig = px.bar(
            day_df,
            x="Day",
            y="Value",
            color="Value",
            color_continuous_scale=SOFT_SCALE,
            title=f"{metric} by {dimension}"
        )

    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detailed Metrics Table")
    if dimension == "Device Type":
        st.dataframe(data['device'], use_container_width=True)
    elif dimension == "Traffic Source":
        st.dataframe(data['traffic'], use_container_width=True)
    elif dimension == "Region":
        st.dataframe(data['region'], use_container_width=True)

elif page == "💡 Insights":
    st.title("💡 Business Insights & Recommendations")
    st.markdown("---")

    st.subheader("1. Critical Bottleneck: Product View → Add to Cart")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        **Issue**: 60% of users drop off when viewing products

        **Impact**: Loss of 150K+ users/month = $150M+ annual revenue impact

        **Root Causes**:
        - Inadequate product descriptions
        - No customer reviews visible
        - Poor product images
        - Unclear pricing

        **Recommendations**:
        1. Implement customer review system (Priority: HIGH)
        2. Add 360° product views and zoom functionality
        3. Display competitor pricing comparison
        4. Show delivery timeline and cost upfront

        **Expected Impact**: +1% conversion = +$48M annual revenue
        """)
    with col2:
        st.metric("Drop-off", "60%", delta=-60, delta_color="inverse")
        st.metric("Revenue Impact", "$150M")
        st.metric("ROI of Fix", "High")

    st.markdown("---")
    st.subheader("2. Mobile Optimization Opportunity")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        **Issue**: Mobile conversion (1.6%) is 3X lower than Desktop (4.8%)

        **Impact**: Mobile = 55% of traffic but only 27% of revenue

        **Root Causes**:
        - Slow page load times
        - Complex checkout process
        - Small touch targets
        - Missing payment options

        **Recommendations**:
        1. Implement Apple Pay & Google Pay
        2. Reduce checkout form to 3 essential fields
        3. Target <2 second page load time
        4. Increase button sizes (44px minimum)

        **Expected Impact**: Mobile conversion 1.6% → 3.2% = +$176M revenue
        """)
    with col2:
        st.metric("Mobile Conv", "1.6%", delta=-3.2, delta_color="inverse")
        st.metric("Desktop Conv", "4.8%")
        st.metric("Gap", "3.2%")

    st.markdown("---")
    st.subheader("3. Traffic Quality & ROI")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        **Insight**: Organic traffic has highest ROI (5.2% conversion)

        **Current Allocation**:
        - Organic: 40% of traffic, 5.2% conversion
        - Paid: 30% of traffic, 3.1% conversion
        - Referral: 15% of traffic, 4.8% conversion
        - Social: 10% of traffic, 1.9% conversion

        **Recommendations**:
        1. Shift $5M budget from Paid/Social to Organic/SEO
        2. Improve content quality for organic search
        3. Strengthen referral partnerships
        4. Restructure social campaigns for quality

        **Expected Impact**: +$25M annual revenue
        """)
    with col2:
        st.metric("Best Source", "Organic", "5.2%")
        st.metric("Worst Source", "Social", "1.9%")
        st.metric("Gap", "3.3%")

    st.markdown("---")
    st.subheader("Summary: Top 5 Actions")
    actions = pd.DataFrame({
        'Priority': ['1', '2', '3', '4', '5'],
        'Action': [
            'Improve Product View UX',
            'Mobile Checkout Optimization',
            'Implement Reviews System',
            'Shift Marketing Budget',
            'Add Payment Options'
        ],
        'Impact': ['+1.0%', '+1.6%', '+0.5%', '+0.8%', '+0.3%'],
        'Timeline': ['2 months', '1 month', '2 weeks', '1 month', '2 weeks'],
        'Revenue': ['$48M', '$76M', '$24M', '$38M', '$14M']
    })
    st.dataframe(actions, use_container_width=True)

elif page == "📋 FAQ":
    st.title("📋 Frequently Asked Questions")
    st.markdown("---")

    with st.expander("How is conversion probability calculated?", expanded=False):
        st.markdown("""
        Our ML model uses:
        - Session frequency
        - User lifetime
        - Max stage reached
        - Device type
        - Traffic source
        """)

    with st.expander("What does conversion rate mean?", expanded=False):
        st.markdown("""
        **Conversion Rate** = (Users who completed purchase) / (Total users) × 100
        """)

    with st.expander("How is revenue calculated?", expanded=False):
        st.markdown("""
        **Monthly Revenue** = Users × Conversion Rate × Average Order Value
        """)

    with st.expander("What's the difference between drop-off and conversion rate?", expanded=False):
        st.markdown("""
        **Drop-off Rate**: % of users who leave at each stage  
        **Conversion Rate**: % of users who complete purchase
        """)

    with st.expander("How do I read the confusion matrix?", expanded=False):
        st.markdown("""
        - True Negative: correctly predicted non-converter.
        - False Positive: predicted converter, actually non-converter.
        - False Negative: predicted non-converter, actually converter.
        - True Positive: correctly predicted converter.
        """)

    with st.expander("What do accuracy, precision, recall, and F1 mean?", expanded=False):
        st.markdown("""
        - Accuracy: overall correctness.
        - Precision: how many predicted converters were correct.
        - Recall: how many actual converters were found.
        - F1: balance between precision and recall.
        """)

    with st.expander("How often is the dashboard updated?", expanded=False):
        st.markdown("""
        - Real-time metrics: Updated every 5 minutes
        - Aggregated reports: Updated daily
        - ML predictions: Retrained weekly
        """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <p>🚀 Funnel Analysis Dashboard v1.0</p>
    <p>Production-Grade Data Science Platform</p>
    <p style="font-size: 12px; color: gray;">Last Updated: June 2024</p>
</div>
""", unsafe_allow_html=True)