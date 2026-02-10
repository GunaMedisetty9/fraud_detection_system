# pages/6_📉_ROC_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from utils import COLORS

st.set_page_config(page_title="ROC Analysis", page_icon="📉", layout="wide")

st.title("📉 ROC & Precision-Recall Analysis")
st.markdown("Receiver Operating Characteristic and Precision-Recall curves for model evaluation")

try:
    predictions_df = pd.read_csv('models/predictions.csv')
except:
    st.error("⚠️ Please run model_training.py first")
    st.stop()

models = {
    'Logistic Regression': ('LR_Pred', 'LR_Prob'),
    'Random Forest': ('RF_Pred', 'RF_Prob'),
    'XGBoost': ('XGB_Pred', 'XGB_Prob'),
    'Gradient Boosting': ('GB_Pred', 'GB_Prob')
}

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

y_true = predictions_df['Actual']

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📈 ROC Curves Comparison")
    
    fig = go.Figure()
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', width=2, dash='dash')
    ))
    
    for i, (name, (pred_col, prob_col)) in enumerate(models.items()):
        y_prob = predictions_df[prob_col]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{name} (AUC = {roc_auc:.4f})',
            line=dict(color=colors[i], width=2)
        ))
    
    fig.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=500,
        legend=dict(x=0.6, y=0.1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### 📈 Precision-Recall Curves")
    
    fig = go.Figure()
    
    for i, (name, (pred_col, prob_col)) in enumerate(models.items()):
        y_prob = predictions_df[prob_col]
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name=f'{name} (AUC = {pr_auc:.4f})',
            line=dict(color=colors[i], width=2)
        ))
    
    fig.update_layout(
        xaxis_title='Recall',
        yaxis_title='Precision',
        height=500,
        legend=dict(x=0.6, y=0.9)
    )
    
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Threshold Analysis
st.markdown("### 🎚️ Threshold Analysis")

selected_model = st.selectbox("Select Model for Threshold Analysis", list(models.keys()))
pred_col, prob_col = models[selected_model]
y_prob = predictions_df[prob_col]

thresholds = np.arange(0.1, 1.0, 0.1)
threshold_metrics = []

for thresh in thresholds:
    y_pred_thresh = (y_prob >= thresh).astype(int)
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    threshold_metrics.append({
        'Threshold': thresh,
        'Precision': precision_score(y_true, y_pred_thresh, zero_division=0),
        'Recall': recall_score(y_true, y_pred_thresh, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred_thresh, zero_division=0)
    })

thresh_df = pd.DataFrame(threshold_metrics)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=thresh_df['Threshold'], y=thresh_df['Precision'],
    mode='lines+markers', name='Precision',
    line=dict(color=COLORS['primary'], width=2)
))

fig.add_trace(go.Scatter(
    x=thresh_df['Threshold'], y=thresh_df['Recall'],
    mode='lines+markers', name='Recall',
    line=dict(color=COLORS['success'], width=2)
))

fig.add_trace(go.Scatter(
    x=thresh_df['Threshold'], y=thresh_df['F1 Score'],
    mode='lines+markers', name='F1 Score',
    line=dict(color=COLORS['danger'], width=2)
))

fig.update_layout(
    title=f'{selected_model} - Metrics vs Threshold',
    xaxis_title='Classification Threshold',
    yaxis_title='Score',
    height=400
)

st.plotly_chart(fig, use_container_width=True)