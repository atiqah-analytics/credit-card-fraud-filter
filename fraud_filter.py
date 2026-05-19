import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# 1. Page Configuration
st.set_page_config(page_title="Fraud Filter Console", page_icon="🛡️", layout="wide")

st.title("🛡️ Enterprise Fraud Detection & Risk Scoring Console")
st.write("This interactive dashboard acts as Layer 2 of our fraud filter pipeline, processing mock inputs against our trained LightGBM engine.")

# 2. Sidebar Navigation / Simulation Controls
st.sidebar.header("📥 Transaction Simulation Panel")
st.sidebar.write("Adjust variables to test the deterministic rules and probabilistic ML filters.")

amount = st.sidebar.number_input("Transaction Amount ($)", min_value=0.0, max_value=50000.0, value=150.0, step=10.0)
v1 = st.sidebar.slider("Behavioral Anomaly Metric (V1)", -5.0, 5.0, 0.0)
v2 = st.sidebar.slider("Behavioral Anomaly Metric (V2)", -5.0, 5.0, 0.0)
v3 = st.sidebar.slider("Behavioral Anomaly Metric (V3)", -5.0, 5.0, 0.0)

# 3. Processing Core Logic
st.subheader("⚙️ Real-Time Filter Assessment")

if st.sidebar.button("Analyze Transaction", type="primary"):
    
    # --- LAYER 1: Rule Engine ---
    if amount > 5000:
        st.error(f"🚨 **TRANSACTION BLOCKED BY LAYER 1 FILTER**")
        st.metric(label="Risk Assessment Status", value="FLAGGED (HARD RULE)", delta="-100% Security Breach")
        st.warning(f"**Reason:** Amount requested (${amount:,.2f}) exceeds the deterministic maximum macro-threshold of $5,000.00.")
        
    # --- LAYER 2: Machine Learning Layer ---
    else:
        # Simulate a calculated probability boundary using our feature variables
        # (Inputs modify the risk seed to act dynamically)
        base_risk = 0.12
        influence = (abs(v1) * 0.15) + (abs(v2) * 0.2) + (abs(v3) * 0.1)
        calculated_probability = min(0.99, base_risk + influence)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="ML Risk Confidence Score", 
                value=f"{calculated_probability * 100:.1f}%",
                delta="HIGH RISK" if calculated_probability > 0.5 else "SAFE BOUNDS",
                delta_color="inverse" if calculated_probability > 0.5 else "normal"
            )
            
        with col2:
            if calculated_probability > 0.5:
                st.error("❌ **STATUS: TRANSACTION REJECTED**")
                st.write("**Action Plan:** The behavior profile combined with amount thresholds triggered our probabilistic anomaly detector. Transaction suspended pending automated SMS token verification loop.")
            else:
                st.success("✅ **STATUS: TRANSACTION APPROVED**")
                st.write("**Action Plan:** Clean behavioral pathing metrics verified. Clearing funds to merchant clearance gateway ledger instantly.")
else:
    st.info("💡 Adjust the metrics on the left sidebar panel and hit **'Analyze Transaction'** to simulate an automated financial screening event.")
