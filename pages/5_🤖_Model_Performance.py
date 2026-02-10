# pages/5_🤖_Model_Performance.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from utils import COLORS, create_gauge_chart, create_confusion_matrix_plot

st.set_page_config(page_title="Model Performance", page_icon="🤖", layout="wide")

st.title("🤖 Model Performance")
st.markdown("Comparison of machine learning models for fraud detection")

# Load predictions
try:
    predictions_df = pd.read_csv('models/predictions.csv')
except:
    st.error("⚠️ Please run model_training.py first to generate model predictions")
    st.stop()

# Model names and their prediction columns
models = {
    'Logistic Regression': ('LR_Pred', 'LR_Prob'),
    'Random Forest': ('RF_Pred', 'RF_Prob'),
    'XGBoost': ('XGB_Pred', 'XGB_Prob'),
    'Gradient Boosting': ('GB_Pred', 'GB_Prob')
}

# Calculate metrics for each model
metrics_data = []

for name, (pred_col, prob_col) in models.items():
    y_true = predictions_df['Actual']
    y_pred = predictions_df[pred_col]
    y_prob = predictions_df[prob_col]
    
    metrics_data.append({
        'Model': name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred),
        'ROC AUC': roc_auc_score(y_true, y_prob)
    })

metrics_df = pd.DataFrame(metrics_data)

# Model selector
st.sidebar.header("🔧 Select Model")
selected_model = st.sidebar.selectbox("Choose a model", list(models.keys()))

# Display metrics comparison
st.markdown("### 📊 Model Comparison")

# Create comparison chart
fig = go.Figure()

metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
colors = [COLORS['primary'], COLORS['secondary'], COLORS['success'], COLORS['danger']]

for i, model in enumerate(models.keys()):
    model_metrics = metrics_df[metrics_df['Model'] == model]
    
    fig.add_trace(go.Bar(
        name=model,
        x=metrics_list,
        y=[model_metrics[m].values[0] for m in metrics_list],
        marker_color=colors[i],
        text=[f"{model_metrics[m].values[0]:.4f}" for m in metrics_list],
        textposition='auto'
    ))

fig.update_layout(
    barmode='group',
    xaxis_title='Metric',
    yaxis_title='Score',
    height=500,
    yaxis=dict(range=[0, 1.1])
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Detailed metrics table
st.markdown("### 📋 Detailed Metrics Table")
styled_df = metrics_df.set_index('Model').style.format("{:.4f}").background_gradient(cmap='RdYlGn', axis=0)
st.dataframe(styled_df, use_container_width=True)

st.markdown("---")

# Selected model details
st.markdown(f"### 🎯 {selected_model} - Detailed Analysis")

pred_col, prob_col = models[selected_model]
y_true = predictions_df['Actual']
y_pred = predictions_df[pred_col]
y_prob = predictions_df[prob_col]

col1, col2, col3, col4, col5 = st.columns(5)

model_metrics = metrics_df[metrics_df['Model'] == selected_model].iloc[0]

with col1:
    fig = create_gauge_chart(model_metrics['Accuracy'], 'Accuracy')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = create_gauge_chart(model_metrics['Precision'], 'Precision')
    st.plotly_chart(fig, use_container_width=True)

with col3:
    fig = create_gauge_chart(model_metrics['Recall'], 'Recall')
    st.plotly_chart(fig, use_container_width=True)

with col4:
    fig = create_gauge_chart(model_metrics['F1 Score'], 'F1 Score')
    st.plotly_chart(fig, use_container_width=True)

with col5:
    fig = create_gauge_chart(model_metrics['ROC AUC'], 'ROC AUC')
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Confusion Matrix
st.markdown(f"### 🎯 {selected_model} - Confusion Matrix")
fig = create_confusion_matrix_plot(y_true, y_pred)
st.plotly_chart(fig, use_container_width=True)