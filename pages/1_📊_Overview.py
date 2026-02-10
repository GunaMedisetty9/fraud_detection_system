# pages/1_📊_Overview.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import COLORS, load_data, format_currency

st.set_page_config(page_title="Overview", page_icon="📊", layout="wide")

st.title("📊 Dashboard Overview")
st.markdown("Comprehensive view of transaction data and fraud statistics")

@st.cache_data
def get_data():
    return load_data()

try:
    df = get_data()
except:
    st.error("Please ensure dataset is loaded properly")
    st.stop()

# Date filter simulation (since dataset doesn't have dates)
st.sidebar.header("🔧 Filters")
amount_range = st.sidebar.slider(
    "Transaction Amount Range ($)",
    float(df['Amount'].min()),
    float(df['Amount'].max()),
    (float(df['Amount'].min()), float(df['Amount'].max()))
)

# Filter data
filtered_df = df[(df['Amount'] >= amount_range[0]) & (df['Amount'] <= amount_range[1])]

# Summary Statistics Cards
st.markdown("### 📈 Summary Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Transactions", f"{len(filtered_df):,}")

with col2:
    fraud_rate = (filtered_df['Class'].sum() / len(filtered_df)) * 100
    st.metric("Fraud Rate", f"{fraud_rate:.4f}%")

with col3:
    st.metric("Avg Amount", format_currency(filtered_df['Amount'].mean()))

with col4:
    st.metric("Max Amount", format_currency(filtered_df['Amount'].max()))

st.markdown("---")

# Row 1: Main Charts
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🎯 Class Distribution")
    
    class_counts = filtered_df['Class'].value_counts()
    
    fig = go.Figure(data=[go.Bar(
        x=['Legitimate', 'Fraud'],
        y=[class_counts[0], class_counts[1]],
        marker_color=[COLORS['success'], COLORS['danger']],
        text=[class_counts[0], class_counts[1]],
        textposition='auto'
    )])
    
    fig.update_layout(
        xaxis_title="Transaction Type",
        yaxis_title="Count",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### 💰 Amount by Class (Box Plot)")
    
    fig = go.Figure()
    
    fig.add_trace(go.Box(
        y=filtered_df[filtered_df['Class'] == 0]['Amount'],
        name='Legitimate',
        marker_color=COLORS['success']
    ))
    
    fig.add_trace(go.Box(
        y=filtered_df[filtered_df['Class'] == 1]['Amount'],
        name='Fraud',
        marker_color=COLORS['danger']
    ))
    
    fig.update_layout(
        yaxis_title="Amount ($)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Row 2: Time Analysis
st.markdown("### ⏱️ Transaction Time Analysis")

col1, col2 = st.columns(2)

with col1:
    # Convert time to hours (simulating time of day)
    filtered_df['Hour'] = (filtered_df['Time'] % 86400) / 3600
    
    fig = px.histogram(
        filtered_df, 
        x='Hour', 
        color='Class',
        nbins=24,
        color_discrete_map={0: COLORS['success'], 1: COLORS['danger']},
        labels={'Class': 'Transaction Type'},
        title='Transactions by Hour of Day'
    )
    
    fig.update_layout(
        xaxis_title="Hour of Day",
        yaxis_title="Count",
        height=400,
        barmode='overlay'
    )
    
    fig.for_each_trace(lambda t: t.update(name='Legitimate' if t.name == '0' else 'Fraud'))
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Cumulative transactions over time
    time_df = filtered_df.groupby('Class').apply(
        lambda x: x.sort_values('Time').reset_index(drop=True)
    ).reset_index(drop=True)
    
    fig = go.Figure()
    
    for class_val, color, name in [(0, COLORS['success'], 'Legitimate'), (1, COLORS['danger'], 'Fraud')]:
        class_data = filtered_df[filtered_df['Class'] == class_val].sort_values('Time')
        class_data['Cumulative'] = range(1, len(class_data) + 1)
        
        fig.add_trace(go.Scatter(
            x=class_data['Time'],
            y=class_data['Cumulative'],
            mode='lines',
            name=name,
            line=dict(color=color, width=2)
        ))
    
    fig.update_layout(
        title='Cumulative Transactions Over Time',
        xaxis_title="Time (seconds)",
        yaxis_title="Cumulative Count",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Data Sample
st.markdown("### 📋 Data Sample")
st.dataframe(
    filtered_df.head(100).style.format({
        'Amount': '${:.2f}',
        'Time': '{:.0f}'
    }),
    use_container_width=True
)