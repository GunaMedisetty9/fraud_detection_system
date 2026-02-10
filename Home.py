# Home.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import COLORS, load_data, format_currency, format_percentage

# Page configuration
st.set_page_config(
    page_title="Financial Fraud Detection System",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .feature-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🔐 Financial Fraud Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Real-Time Transaction Monitoring & Fraud Detection</p>', unsafe_allow_html=True)

# Load data
@st.cache_data
def get_data():
    return load_data()

try:
    df = get_data()
    data_loaded = True
except:
    st.error("⚠️ Please ensure the dataset is placed in the 'data/' folder")
    data_loaded = False

if data_loaded:
    # Key Metrics Row
    st.markdown("### 📊 Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_transactions = len(df)
    fraud_cases = df['Class'].sum()
    legitimate_cases = total_transactions - fraud_cases
    fraud_rate = (fraud_cases / total_transactions) * 100
    total_amount = df['Amount'].sum()
    
    with col1:
        st.metric(
            label="📈 Total Transactions",
            value=f"{total_transactions:,}",
            delta="Live Data"
        )
    
    with col2:
        st.metric(
            label="✅ Legitimate",
            value=f"{legitimate_cases:,}",
            delta=f"{100-fraud_rate:.2f}%"
        )
    
    with col3:
        st.metric(
            label="🚨 Fraud Cases",
            value=f"{fraud_cases:,}",
            delta=f"{fraud_rate:.4f}%",
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            label="💰 Total Amount",
            value=format_currency(total_amount),
            delta="Processed"
        )
    
    with col5:
        avg_amount = df['Amount'].mean()
        st.metric(
            label="💵 Avg Transaction",
            value=format_currency(avg_amount),
            delta="Per Transaction"
        )
    
    st.markdown("---")
    
    # Two column layout for charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🥧 Transaction Distribution")
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Legitimate', 'Fraud'],
            values=[legitimate_cases, fraud_cases],
            hole=0.5,
            marker_colors=[COLORS['success'], COLORS['danger']],
            textinfo='label+percent',
            textfont_size=14,
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>"
        )])
        
        fig_pie.update_layout(
            showlegend=True,
            height=400,
            annotations=[dict(text='Transactions', x=0.5, y=0.5, font_size=16, showarrow=False)]
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Transaction Amount Distribution")
        
        fig_hist = go.Figure()
        
        fig_hist.add_trace(go.Histogram(
            x=df[df['Class'] == 0]['Amount'],
            name='Legitimate',
            marker_color=COLORS['success'],
            opacity=0.7,
            nbinsx=50
        ))
        
        fig_hist.add_trace(go.Histogram(
            x=df[df['Class'] == 1]['Amount'],
            name='Fraud',
            marker_color=COLORS['danger'],
            opacity=0.7,
            nbinsx=50
        ))
        
        fig_hist.update_layout(
            barmode='overlay',
            xaxis_title='Transaction Amount ($)',
            yaxis_title='Frequency',
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown("---")
    
    # Feature Statistics
    st.markdown("### 📈 Feature Statistics Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Amount statistics by class
        amount_stats = df.groupby('Class')['Amount'].agg(['mean', 'median', 'std', 'min', 'max'])
        amount_stats.index = ['Legitimate', 'Fraud']
        amount_stats.columns = ['Mean', 'Median', 'Std Dev', 'Min', 'Max']
        
        st.markdown("#### 💵 Amount Statistics by Class")
        st.dataframe(amount_stats.style.format("${:.2f}"), use_container_width=True)
    
    with col2:
        # Time statistics
        time_stats = df.groupby('Class')['Time'].agg(['mean', 'median', 'std'])
        time_stats.index = ['Legitimate', 'Fraud']
        time_stats.columns = ['Mean (sec)', 'Median (sec)', 'Std Dev']
        
        st.markdown("#### ⏱️ Time Statistics by Class")
        st.dataframe(time_stats.style.format("{:.2f}"), use_container_width=True)
    
    st.markdown("---")
    
    # System Features
    st.markdown("### 🎯 System Features")
    
    features = [
        ("🤖 Machine Learning Models", "Logistic Regression, Random Forest, XGBoost, Gradient Boosting"),
        ("📊 Interactive Dashboards", "10+ visualization pages with real-time updates"),
        ("🚨 Alert System", "Instant fraud detection alerts with risk scoring"),
        ("📈 Performance Metrics", "Accuracy, Precision, Recall, F1-Score, ROC-AUC"),
        ("🔄 Real-time Prediction", "Instant transaction classification"),
        ("📋 Comprehensive Reports", "Detailed fraud analysis and reporting")
    ]
    
    col1, col2 = st.columns(2)
    
    for i, (title, desc) in enumerate(features):
        with col1 if i % 2 == 0 else col2:
            st.markdown(f"""
            <div class="feature-box">
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>💡 Navigate through different pages using the sidebar to explore all features</p>
        <p>Developed with ❤️ using Python, Streamlit, and Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("📁 Please download the Credit Card Fraud Detection dataset from Kaggle and place it in the 'data/' folder")
    st.markdown("""
    ### Setup Instructions:
    1. Download dataset from: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
    2. Create a `data` folder in the project directory
    3. Place `creditcard.csv` in the `data` folder
    4. Run `python model_training.py` to train models
    5. Restart the Streamlit app
    """)