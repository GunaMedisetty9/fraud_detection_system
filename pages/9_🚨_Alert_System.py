# pages/9_🚨_Alert_System.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from utils import COLORS

st.set_page_config(page_title="Alert System", page_icon="🚨", layout="wide")

st.title("🚨 Fraud Alert System")
st.markdown("Real-time monitoring and alert management")

# Initialize session state for alerts
if 'alerts' not in st.session_state:
    st.session_state.alerts = []

# Alert configuration
st.sidebar.header("⚙️ Alert Configuration")

alert_threshold = st.sidebar.slider("Fraud Probability Threshold", 0.0, 1.0, 0.5)
enable_email = st.sidebar.checkbox("Enable Email Alerts", value=False)
enable_sms = st.sidebar.checkbox("Enable SMS Alerts", value=False)

if enable_email:
    email_address = st.sidebar.text_input("Alert Email", placeholder="your@email.com")

if enable_sms:
    phone_number = st.sidebar.text_input("Phone Number", placeholder="+1234567890")

# Alert Statistics
st.markdown("### 📊 Alert Statistics")

col1, col2, col3, col4 = st.columns(4)

# Generate sample alerts for demonstration
np.random.seed(42)
sample_alerts = []
for i in range(50):
    sample_alerts.append({
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Transaction ID': f"TXN{np.random.randint(100000, 999999)}",
        'Amount': np.random.uniform(10, 5000),
        'Fraud Probability': np.random.uniform(0.5, 0.99),
        'Status': np.random.choice(['Pending', 'Reviewed', 'Confirmed Fraud', 'False Positive']),
        'Risk Level': np.random.choice(['High', 'Medium', 'Critical'])
    })

alerts_df = pd.DataFrame(sample_alerts)

with col1:
    st.metric("Total Alerts", len(alerts_df))

with col2:
    critical = len(alerts_df[alerts_df['Risk Level'] == 'Critical'])
    st.metric("Critical Alerts", critical, delta=f"+{np.random.randint(1, 5)} today")

with col3:
    pending = len(alerts_df[alerts_df['Status'] == 'Pending'])
    st.metric("Pending Review", pending)

with col4:
    confirmed = len(alerts_df[alerts_df['Status'] == 'Confirmed Fraud'])
    st.metric("Confirmed Frauds", confirmed)

st.markdown("---")

# Alert visualization
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📈 Alerts by Risk Level")
    
    risk_counts = alerts_df['Risk Level'].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=risk_counts.index,
        values=risk_counts.values,
        hole=0.4,
        marker_colors=[COLORS['danger'], COLORS['warning'], COLORS['secondary']]
    )])
    
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### 📊 Alert Status Distribution")
    
    status_counts = alerts_df['Status'].value_counts()
    
    fig = go.Figure(data=go.Bar(
        x=status_counts.index,
        y=status_counts.values,
        marker_color=[COLORS['warning'], COLORS['primary'], COLORS['danger'], COLORS['success']]
    ))
    
    fig.update_layout(height=350, xaxis_title="Status", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Recent Alerts Table
st.markdown("### 📋 Recent Fraud Alerts")

# Add filters
col1, col2, col3 = st.columns(3)

with col1:
    status_filter = st.selectbox("Filter by Status", ['All'] + list(alerts_df['Status'].unique()))

with col2:
    risk_filter = st.selectbox("Filter by Risk Level", ['All'] + list(alerts_df['Risk Level'].unique()))

with col3:
    sort_by = st.selectbox("Sort by", ['Fraud Probability', 'Amount', 'Timestamp'])

# Apply filters
filtered_alerts = alerts_df.copy()
if status_filter != 'All':
    filtered_alerts = filtered_alerts[filtered_alerts['Status'] == status_filter]
if risk_filter != 'All':
    filtered_alerts = filtered_alerts[filtered_alerts['Risk Level'] == risk_filter]

filtered_alerts = filtered_alerts.sort_values(sort_by, ascending=False)

# Display alerts with styling
def highlight_risk(row):
    if row['Risk Level'] == 'Critical':
        return ['background-color: #ffcccc'] * len(row)
    elif row['Risk Level'] == 'High':
        return ['background-color: #ffe6cc'] * len(row)
    return [''] * len(row)

styled_alerts = filtered_alerts.head(20).style.apply(highlight_risk, axis=1).format({
    'Amount': '${:.2f}',
    'Fraud Probability': '{:.2%}'
})

st.dataframe(styled_alerts, use_container_width=True)

st.markdown("---")

# Alert Actions
st.markdown("### ⚡ Quick Actions")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📧 Send Summary Email", use_container_width=True):
        st.success("✅ Summary email sent successfully!")

with col2:
    if st.button("📥 Export Alerts CSV", use_container_width=True):
        st.download_button(
            label="Download CSV",
            data=alerts_df.to_csv(index=False),
            file_name="fraud_alerts.csv",
            mime="text/csv"
        )

with col3:
    if st.button("🔄 Refresh Alerts", use_container_width=True):
        st.experimental_rerun()

with col4:
    if st.button("✅ Mark All as Reviewed", use_container_width=True):
        st.success("✅ All alerts marked as reviewed!")