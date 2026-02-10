# pages/2_📈_Transaction_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import COLORS, load_data

st.set_page_config(page_title="Transaction Analysis", page_icon="📈", layout="wide")

st.title("📈 Transaction Analysis")
st.markdown("Deep dive into transaction patterns and behaviors")

@st.cache_data
def get_data():
    return load_data()

df = get_data()

# Sidebar filters
st.sidebar.header("🔧 Analysis Options")
analysis_type = st.sidebar.selectbox(
    "Select Analysis Type",
    ["Amount Distribution", "Feature Correlation", "Transaction Patterns", "Anomaly Detection"]
)

if analysis_type == "Amount Distribution":
    st.markdown("### 💰 Transaction Amount Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram with KDE
        fig = go.Figure()
        
        for class_val, color, name in [(0, COLORS['success'], 'Legitimate'), (1, COLORS['danger'], 'Fraud')]:
            class_data = df[df['Class'] == class_val]['Amount']
            
            fig.add_trace(go.Histogram(
                x=class_data,
                name=name,
                marker_color=color,
                opacity=0.7,
                nbinsx=100
            ))
        
        fig.update_layout(
            title='Amount Distribution by Class',
            xaxis_title='Amount ($)',
            yaxis_title='Frequency',
            barmode='overlay',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Violin Plot
        fig = go.Figure()
        
        fig.add_trace(go.Violin(
            y=df[df['Class'] == 0]['Amount'],
            name='Legitimate',
            box_visible=True,
            meanline_visible=True,
            fillcolor=COLORS['success'],
            opacity=0.6,
            line_color=COLORS['success']
        ))
        
        fig.add_trace(go.Violin(
            y=df[df['Class'] == 1]['Amount'],
            name='Fraud',
            box_visible=True,
            meanline_visible=True,
            fillcolor=COLORS['danger'],
            opacity=0.6,
            line_color=COLORS['danger']
        ))
        
        fig.update_layout(
            title='Amount Violin Plot by Class',
            yaxis_title='Amount ($)',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistics table
    st.markdown("### 📊 Descriptive Statistics")
    
    stats = df.groupby('Class')['Amount'].describe()
    stats.index = ['Legitimate', 'Fraud']
    st.dataframe(stats.style.format("${:.2f}"), use_container_width=True)

elif analysis_type == "Feature Correlation":
    st.markdown("### 🔗 Feature Correlation Analysis")
    
    # Select features for correlation
    features = st.multiselect(
        "Select features for correlation",
        df.columns.tolist()[:-1],
        default=['V1', 'V2', 'V3', 'V4', 'V5', 'Amount']
    )
    
    if features:
        corr_matrix = df[features + ['Class']].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Feature Correlation Heatmap',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "Transaction Patterns":
    st.markdown("### 📊 Transaction Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top features correlated with fraud
        correlations = df.corr()['Class'].drop('Class').sort_values()
        
        fig = go.Figure(data=go.Bar(
            x=correlations.values,
            y=correlations.index,
            orientation='h',
            marker_color=[COLORS['danger'] if x < 0 else COLORS['success'] for x in correlations.values]
        ))
        
        fig.update_layout(
            title='Feature Correlation with Fraud',
            xaxis_title='Correlation Coefficient',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Feature importance for fraud detection
        fraud_means = df[df['Class'] == 1].mean()
        legit_means = df[df['Class'] == 0].mean()
        
        diff = (fraud_means - legit_means).drop('Class').sort_values()
        
        fig = go.Figure(data=go.Bar(
            x=diff.values,
            y=diff.index,
            orientation='h',
            marker_color=[COLORS['danger'] if x < 0 else COLORS['primary'] for x in diff.values]
        ))
        
        fig.update_layout(
            title='Mean Difference (Fraud - Legitimate)',
            xaxis_title='Difference',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)

else:  # Anomaly Detection
    st.markdown("### 🔍 Anomaly Detection Visualization")
    
    feature_x = st.selectbox("X-axis Feature", df.columns[:-1], index=0)
    feature_y = st.selectbox("Y-axis Feature", df.columns[:-1], index=1)
    
    # Sample data for visualization
    sample_df = df.sample(min(5000, len(df)), random_state=42)
    
    fig = px.scatter(
        sample_df,
        x=feature_x,
        y=feature_y,
        color='Class',
        color_discrete_map={0: COLORS['success'], 1: COLORS['danger']},
        opacity=0.6,
        title=f'{feature_x} vs {feature_y} (Sampled Data)'
    )
    
    fig.update_layout(height=600)
    fig.for_each_trace(lambda t: t.update(name='Legitimate' if t.name == '0' else 'Fraud'))
    
    st.plotly_chart(fig, use_container_width=True)