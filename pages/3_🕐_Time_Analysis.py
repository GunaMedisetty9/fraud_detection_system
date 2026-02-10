# pages/3_🕐_Time_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import COLORS, load_data

st.set_page_config(page_title="Time Analysis", page_icon="🕐", layout="wide")

st.title("🕐 Time-Based Analysis")
st.markdown("Understanding temporal patterns in fraudulent transactions")

@st.cache_data
def get_data():
    return load_data()

df = get_data()

# Create time-based features
df['Hour'] = (df['Time'] % 86400) / 3600
df['Day'] = df['Time'] // 86400
df['Hour_Bin'] = pd.cut(df['Hour'], bins=24, labels=range(24))

st.markdown("### ⏰ Hourly Transaction Patterns")

col1, col2 = st.columns(2)

with col1:
    # Transactions by hour
    hourly_data = df.groupby(['Hour_Bin', 'Class']).size().unstack(fill_value=0)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=hourly_data.index.astype(str),
        y=hourly_data[0],
        name='Legitimate',
        marker_color=COLORS['success']
    ))
    
    fig.add_trace(go.Bar(
        x=hourly_data.index.astype(str),
        y=hourly_data[1],
        name='Fraud',
        marker_color=COLORS['danger']
    ))
    
    fig.update_layout(
        title='Transactions by Hour',
        xaxis_title='Hour of Day',
        yaxis_title='Count',
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Fraud rate by hour
    fraud_rate_hourly = df.groupby('Hour_Bin')['Class'].mean() * 100
    
    fig = go.Figure(data=go.Scatter(
        x=fraud_rate_hourly.index.astype(str),
        y=fraud_rate_hourly.values,
        mode='lines+markers',
        line=dict(color=COLORS['danger'], width=3),
        marker=dict(size=10),
        fill='tozeroy',
        fillcolor='rgba(231, 76, 60, 0.3)'
    ))
    
    fig.update_layout(
        title='Fraud Rate by Hour',
        xaxis_title='Hour of Day',
        yaxis_title='Fraud Rate (%)',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Heatmap visualization
st.markdown("### 🗓️ Transaction Heatmap")

# Create heatmap data
df['Day_Num'] = (df['Day'] % 7).astype(int)
heatmap_data = df.groupby(['Day_Num', 'Hour_Bin'])['Class'].mean().unstack(fill_value=0)

fig = go.Figure(data=go.Heatmap(
    z=heatmap_data.values * 100,
    x=[str(i) for i in range(24)],
    y=['Day ' + str(i+1) for i in range(min(7, len(heatmap_data)))],
    colorscale='Reds',
    text=np.round(heatmap_data.values * 100, 2),
    texttemplate='%{text:.2f}%',
    textfont={"size": 9},
    hoverongaps=False
))

fig.update_layout(
    title='Fraud Rate Heatmap (Day vs Hour)',
    xaxis_title='Hour',
    yaxis_title='Day',
    height=400
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Time series analysis
st.markdown("### 📈 Transaction Volume Over Time")

col1, col2 = st.columns(2)

with col1:
    # Daily transaction volume
    daily_volume = df.groupby('Day').size()
    
    fig = go.Figure(data=go.Scatter(
        x=daily_volume.index,
        y=daily_volume.values,
        mode='lines',
        line=dict(color=COLORS['primary'], width=2),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.3)'
    ))
    
    fig.update_layout(
        title='Daily Transaction Volume',
        xaxis_title='Day',
        yaxis_title='Number of Transactions',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Daily fraud count
    daily_fraud = df.groupby('Day')['Class'].sum()
    
    fig = go.Figure(data=go.Scatter(
        x=daily_fraud.index,
        y=daily_fraud.values,
        mode='lines+markers',
        line=dict(color=COLORS['danger'], width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title='Daily Fraud Count',
        xaxis_title='Day',
        yaxis_title='Fraud Cases',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Statistics
st.markdown("### 📊 Time-Based Statistics")

col1, col2, col3 = st.columns(3)

with col1:
    peak_hour = fraud_rate_hourly.idxmax()
    st.metric("Peak Fraud Hour", f"{peak_hour}:00", f"{fraud_rate_hourly.max():.4f}% rate")

with col2:
    safest_hour = fraud_rate_hourly.idxmin()
    st.metric("Safest Hour", f"{safest_hour}:00", f"{fraud_rate_hourly.min():.4f}% rate")

with col3:
    avg_fraud_rate = fraud_rate_hourly.mean()
    st.metric("Average Fraud Rate", f"{avg_fraud_rate:.4f}%")