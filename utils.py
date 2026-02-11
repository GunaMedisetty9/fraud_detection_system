# utils.py
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Color scheme for consistent styling
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ecc71',
    'danger': '#e74c3c',
    'warning': '#f39c12',
    'info': '#3498db',
    'dark': '#2c3e50',
    'light': '#ecf0f1',
    'fraud': '#e74c3c',
    'legitimate': '#2ecc71'
}

from pathlib import Path
import pandas as pd

def load_data():
    full = Path("data/creditcard.csv")
    if full.exists():
        return pd.read_csv(full)

    sample = Path("data/creditcard_sample.csv")
    if sample.exists():
        return pd.read_csv(sample)

    test = Path("models/test_data.csv")
    if test.exists():
        df = pd.read_csv(test)
        return df.rename(columns={"Actual": "Class"})

    raise FileNotFoundError("No dataset found in data/ or models/.")

def get_metrics(y_true, y_pred, y_prob=None):
    """Calculate all evaluation metrics"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred),
    }
    if y_prob is not None:
        metrics['ROC AUC'] = roc_auc_score(y_true, y_prob)
    return metrics

def create_gauge_chart(value, title, max_val=1):
    """Create a gauge chart for metrics"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20}},
        gauge={
            'axis': {'range': [None, max_val], 'tickwidth': 1},
            'bar': {'color': COLORS['primary']},
            'bgcolor': "white",
            'borderwidth': 2,
            'steps': [
                {'range': [0, max_val*0.5], 'color': COLORS['danger']},
                {'range': [max_val*0.5, max_val*0.75], 'color': COLORS['warning']},
                {'range': [max_val*0.75, max_val], 'color': COLORS['success']}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def create_confusion_matrix_plot(y_true, y_pred):
    """Create interactive confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted Legitimate', 'Predicted Fraud'],
        y=['Actual Legitimate', 'Actual Fraud'],
        colorscale='RdYlGn_r',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 20},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='Actual Label',
        height=400
    )
    return fig

def create_roc_curve(y_true, y_prob, model_name):
    """Create ROC curve plot"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'{model_name} (AUC = {auc:.4f})',
        line=dict(color=COLORS['primary'], width=2)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=f'ROC Curve - {model_name}',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=500,
        showlegend=True
    )
    return fig

def format_currency(value):
    """Format number as currency"""
    return f"${value:,.2f}"

def format_percentage(value):
    """Format number as percentage"""
    return f"{value:.2f}%"

def get_risk_level(probability):
    """Determine risk level based on fraud probability"""
    if probability < 0.3:
        return "Low Risk", COLORS['success']
    elif probability < 0.7:
        return "Medium Risk", COLORS['warning']
    else:
        return "High Risk", COLORS['danger']