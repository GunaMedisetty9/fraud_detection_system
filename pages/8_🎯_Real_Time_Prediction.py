# pages/8_🎯_Real_Time_Prediction.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from utils import COLORS, get_risk_level, load_data

st.set_page_config(page_title="Real-Time Prediction", page_icon="🎯", layout="wide")

st.title("🎯 Real-Time Fraud Prediction")
st.markdown("Enter transaction details to predict if it's fraudulent")

# Load models
try:
    models = {
        'Logistic Regression': joblib.load('models/logistic_regression.pkl'),
        'Random Forest': joblib.load('models/random_forest.pkl'),
        'XGBoost': joblib.load('models/xgboost.pkl'),
        'Gradient Boosting': joblib.load('models/gradient_boosting.pkl')
    }
    scaler = joblib.load('models/scaler.pkl')
    df = load_data()
except Exception as e:
    st.error("⚠️ Model loading failed (showing actual error below).")
    st.exception(e)

    import os
    from pathlib import Path
    st.write("CWD:", Path.cwd())
    st.write("models exists:", Path("models").exists())
    if Path("models").exists():
        st.write("models files:", os.listdir("models"))

    st.stop()

# Model selection
selected_model_name = st.sidebar.selectbox("Select Model", list(models.keys()))
selected_model = models[selected_model_name]

st.markdown("### 📝 Enter Transaction Details")

# Create input form
col1, col2, col3 = st.columns(3)

with col1:
    time = st.number_input("Time (seconds)", min_value=0.0, value=0.0)
    amount = st.number_input("Amount ($)", min_value=0.0, value=100.0)
    v1 = st.number_input("V1", value=0.0, format="%.4f")
    v2 = st.number_input("V2", value=0.0, format="%.4f")
    v3 = st.number_input("V3", value=0.0, format="%.4f")
    v4 = st.number_input("V4", value=0.0, format="%.4f")
    v5 = st.number_input("V5", value=0.0, format="%.4f")
    v6 = st.number_input("V6", value=0.0, format="%.4f")
    v7 = st.number_input("V7", value=0.0, format="%.4f")
    v8 = st.number_input("V8", value=0.0, format="%.4f")

with col2:
    v9 = st.number_input("V9", value=0.0, format="%.4f")
    v10 = st.number_input("V10", value=0.0, format="%.4f")
    v11 = st.number_input("V11", value=0.0, format="%.4f")
    v12 = st.number_input("V12", value=0.0, format="%.4f")
    v13 = st.number_input("V13", value=0.0, format="%.4f")
    v14 = st.number_input("V14", value=0.0, format="%.4f")
    v15 = st.number_input("V15", value=0.0, format="%.4f")
    v16 = st.number_input("V16", value=0.0, format="%.4f")
    v17 = st.number_input("V17", value=0.0, format="%.4f")
    v18 = st.number_input("V18", value=0.0, format="%.4f")

with col3:
    v19 = st.number_input("V19", value=0.0, format="%.4f")
    v20 = st.number_input("V20", value=0.0, format="%.4f")
    v21 = st.number_input("V21", value=0.0, format="%.4f")
    v22 = st.number_input("V22", value=0.0, format="%.4f")
    v23 = st.number_input("V23", value=0.0, format="%.4f")
    v24 = st.number_input("V24", value=0.0, format="%.4f")
    v25 = st.number_input("V25", value=0.0, format="%.4f")
    v26 = st.number_input("V26", value=0.0, format="%.4f")
    v27 = st.number_input("V27", value=0.0, format="%.4f")
    v28 = st.number_input("V28", value=0.0, format="%.4f")

# Random transaction button
if st.button("🎲 Load Random Transaction"):
    random_idx = np.random.randint(0, len(df))
    st.info(f"Loaded transaction #{random_idx} - Actual Class: {'Fraud' if df.iloc[random_idx]['Class'] == 1 else 'Legitimate'}")

# Predict button
if st.button("🔮 Predict Transaction", type="primary"):
    # Create feature vector
    features = [time, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, 
                v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, amount]
    
    features_array = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features_array)
    
    # Get prediction
    prediction = selected_model.predict(features_scaled)[0]
    probability = selected_model.predict_proba(features_scaled)[0]
    
    fraud_prob = probability[1]
    risk_level, risk_color = get_risk_level(fraud_prob)
    
    st.markdown("---")
    st.markdown("### 🎯 Prediction Result")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if prediction == 1:
            st.error("## 🚨 FRAUD DETECTED!")
        else:
            st.success("## ✅ LEGITIMATE TRANSACTION")
    
    with col2:
        st.markdown(f"### Fraud Probability: {fraud_prob:.2%}")
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fraud_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': risk_color},
                'steps': [
                    {'range': [0, 30], 'color': 'lightgreen'},
                    {'range': [30, 70], 'color': 'yellow'},
                    {'range': [70, 100], 'color': 'salmon'}
                ]
            }
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.markdown(f"### Risk Level")
        st.markdown(f"<h2 style='color: {risk_color};'>{risk_level}</h2>", unsafe_allow_html=True)
        st.metric("Model Used", selected_model_name)
    
    # All models comparison
    st.markdown("### 📊 All Models Comparison")
    
    all_predictions = []
    for name, model in models.items():
        pred = model.predict(features_scaled)[0]
        prob = model.predict_proba(features_scaled)[0][1]
        all_predictions.append({
            'Model': name,
            'Prediction': 'Fraud' if pred == 1 else 'Legitimate',
            'Fraud Probability': f"{prob:.2%}"
        })
    
    st.dataframe(pd.DataFrame(all_predictions), use_container_width=True)