# pages/4_💳_Amount_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import COLORS, load_data, format_currency

st.set_page_config(page_title="Amount Analysis", page_icon="💳", layout="wide")

st.title("💳 Amount Analysis")
st.markdown("Detailed analysis of transaction amounts")

@st.cache_data
def get_data():
    return load_data()

df = get_data()

# Amount bins
df['Amount_Bin'] = pd.cut(df['Amount'], 
                          bins=[0, 10, 50, 100, 500, 1000, float('inf')],
                          labels=['$0-10', '$10-50', '$50-100', '$100-500', '$500-1000', '$1000+'])

st.markdown("### 💵 Amount Distribution Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Mean Amount", format_currency(df['Amount'].mean()))

with col2:
    st.metric("Median Amount", format_currency(df['Amount'].median()))

with col3:
    st.metric("Max Amount", format_currency(df['Amount'].max()))

with col4:
    st.metric("Std Deviation", format_currency(df['Amount'].std()))

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Transaction Count by Amount Range")
    
    amount_counts = df.groupby(['Amount_Bin', 'Class']).size().unstack(fill_value=0)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=amount_counts.index.astype(str),
        y=amount_counts[0],
        name='Legitimate',
        marker_color=COLORS['success']
    ))
    
    fig.add_trace(go.Bar(
        x=amount_counts.index.astype(str),
        y=amount_counts[1],
        name='Fraud',
        marker_color=COLORS['danger']
    ))
    
    fig.update_layout(
        barmode='group',
        xaxis_title='Amount Range',
        yaxis_title='Count',
        height=450
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### 🎯 Fraud Rate by Amount Range")
    
    fraud_rate = df.groupby('Amount_Bin')['Class'].mean() * 100
    
    fig = go.Figure(data=go.Bar(
        x=fraud_rate.index.astype(str),
        y=fraud_rate.values,
        marker_color=[COLORS['danger'] if x > fraud_rate.mean() else COLORS['warning'] for x in fraud_rate.values],
        text=[f'{x:.4f}%' for x in fraud_rate.values],
        textposition='auto'
    ))
    
    # Add average line
    fig.add_hline(y=fraud_rate.mean(), line_dash="dash", line_color="red",
                  annotation_text=f"Avg: {fraud_rate.mean():.4f}%")
    
    fig.update_layout(
        xaxis_title='Amount Range',
        yaxis_title='Fraud Rate (%)',
        height=450
    )
    
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Detailed comparison
st.markdown("### 📈 Fraud vs Legitimate Amount Comparison")

col1, col2 = st.columns(2)

with col1:
    # CDF plot
    legit_amounts = np.sort(df[df['Class'] == 0]['Amount'])
    fraud_amounts = np.sort(df[df['Class'] == 1]['Amount'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=legit_amounts,
        y=np.arange(1, len(legit_amounts) + 1) / len(legit_amounts),
        mode='lines',
        name='Legitimate',
        line=dict(color=COLORS['success'], width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=fraud_amounts,
        y=np.arange(1, len(fraud_amounts) + 1) / len(fraud_amounts),
        mode='lines',
        name='Fraud',
        line=dict(color=COLORS['danger'], width=2)
    ))
    
    fig.update_layout(
        title='Cumulative Distribution Function (CDF)',
        xaxis_title='Amount ($)',
        yaxis_title='Cumulative Probability',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Log-scale histogram
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=np.log1p(df[df['Class'] == 0]['Amount']),
        name='Legitimate',
        marker_color=COLORS['success'],
        opacity=0.7
    ))
    
    fig.add_trace(go.Histogram(
        x=np.log1p(df[df['Class'] == 1]['Amount']),
        name='Fraud',
        marker_color=COLORS['danger'],
        opacity=0.7
    ))
    
    fig.update_layout(
        title='Log-Transformed Amount Distribution',
        xaxis_title='Log(Amount + 1)',
        yaxis_title='Frequency',
        barmode='overlay',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Statistics table
st.markdown("### 📋 Detailed Statistics by Amount Range")

stats_df = df.groupby('Amount_Bin').agg({
    'Amount': ['count', 'mean', 'std', 'min', 'max'],
    'Class': ['sum', 'mean']
}).round(2)

stats_df.columns = ['Count', 'Mean Amount', 'Std Dev', 'Min', 'Max', 'Fraud Count', 'Fraud Rate']
stats_df['Fraud Rate'] = (stats_df['Fraud Rate'] * 100).round(4).astype(str) + '%'

st.dataframe(stats_df, use_container_width=True)