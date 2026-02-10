# pages/7_🔍_Feature_Importance.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from utils import COLORS, load_data

st.set_page_config(page_title="Feature Importance", page_icon="🔍", layout="wide")

st.title("🔍 Feature Importance Analysis")
st.markdown("Understanding which features contribute most to fraud detection")

# Load model and data
try:
    rf_model = joblib.load('models/random_forest.pkl')
    xgb_model = joblib.load('models/xgboost.pkl')
    df = load_data()
except:
    st.error("⚠️ Please run model_training.py first")
    st.stop()

feature_names = df.columns[:-1].tolist()

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🌲 Random Forest Feature Importance")
    
    rf_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    fig = go.Figure(data=go.Bar(
        x=rf_importance['Importance'],
        y=rf_importance['Feature'],
        orientation='h',
        marker_color=COLORS['primary']
    ))
    
    fig.update_layout(
        xaxis_title='Importance',
        height=700
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### 🚀 XGBoost Feature Importance")
    
    xgb_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': xgb_model.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    fig = go.Figure(data=go.Bar(
        x=xgb_importance['Importance'],
        y=xgb_importance['Feature'],
        orientation='h',
        marker_color=COLORS['secondary']
    ))
    
    fig.update_layout(
        xaxis_title='Importance',
        height=700
    )
    
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Top features comparison
st.markdown("### 🏆 Top 10 Most Important Features")

top_rf = rf_importance.tail(10).set_index('Feature')
top_xgb = xgb_importance.tail(10).set_index('Feature')

comparison_df = pd.DataFrame({
    'Random Forest': top_rf['Importance'],
    'XGBoost': top_xgb['Importance']
}).fillna(0)

fig = go.Figure()

fig.add_trace(go.Bar(
    name='Random Forest',
    x=comparison_df.index,
    y=comparison_df['Random Forest'],
    marker_color=COLORS['primary']
))

fig.add_trace(go.Bar(
    name='XGBoost',
    x=comparison_df.index,
    y=comparison_df['XGBoost'],
    marker_color=COLORS['secondary']
))

fig.update_layout(
    barmode='group',
    xaxis_title='Feature',
    yaxis_title='Importance',
    height=400
)

st.plotly_chart(fig, use_container_width=True)

# Feature correlation with target
st.markdown("### 🔗 Feature Correlation with Fraud")

correlations = df.corr()['Class'].drop('Class').sort_values()

fig = go.Figure(data=go.Bar(
    x=correlations.values,
    y=correlations.index,
    orientation='h',
    marker_color=[COLORS['success'] if x < 0 else COLORS['danger'] for x in correlations.values]
))

fig.update_layout(
    title='Correlation with Fraud (Class)',
    xaxis_title='Correlation Coefficient',
    height=700
)

st.plotly_chart(fig, use_container_width=True)