# ====================================================
#   MODEL PERFORMANCE & DEEP ANALYSIS DASHBOARD
#   Advanced Insights into the AI Brain Behind BTC Prediction
#   Author: Zain308 | Status: PRODUCTION-GRADE MLOPS VIEW
# ====================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Safe imports for FinBERT
try:
    from src.ml.sentiment import get_sentiment, load_sentiment_model
    SENTIMENT_AVAILABLE = True
except Exception as e:
    SENTIMENT_AVAILABLE = False
    st.warning("FinBERT model not loaded – running in demo mode")

# ====================================================
# PAGE CONFIG & STYLING
# ====================================================
st.set_page_config(
    page_title="AI Model Analysis – BTC Predictor",
    page_icon="Brain",
    layout="wide"
)

# Custom CSS for pro look
st.markdown("""
<style>
    .big-font {font-size: 50px !important; font-weight: bold; text-align: center; color: #00D4FF;}
    .metric-card {background: linear-gradient(90deg, #1e1e2e, #2a2a40); padding: 1.5rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,212,255,0.2);}
    .header-box {background: linear-gradient(120deg, #007BFF, #00D4FF); padding: 2rem; border-radius: 20px; text-align: center; color: white; margin-bottom: 2rem;}
    .stApp {background-color: #0e1117;}
</style>
""", unsafe_allow_html=True)

# ====================================================
# HERO HEADER
# ====================================================
st.markdown(f"""
<div class="header-box">
    <h1>AI Model Deep Dive & Performance Analysis</h1>
    <p style="font-size: 20px; opacity: 0.9;">
        Inside the Neural Network that predicts Bitcoin's next move<br>
        Stacked LSTM + FinBERT Sentiment • 60-Hour Memory • Real-Time Inference
    </p>
    <h3>Current Model Version: <span style="color:#00ff88">v2.3-pro</span> • Trained on 2+ Years of Kraken Data</h3>
</div>
""", unsafe_allow_html=True)

# ====================================================
# SECTION 1: FINBERT SENTIMENT MODEL
# ====================================================
st.markdown("---")
st.markdown("<h2 style='text-align: center; color: #00D4FF;'>Sentiment Engine – FinBERT (Financial BERT)</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div class="metric-card">
        <h3 style='text-align: center; color: #00ff88;'>ProsusAI/finbert</h3>
        <p style='text-align: center; font-size: 18px;'>
            Trained on 10,000+ financial news headlines<br>
            Accuracy: <b>~89%</b> on financial sentiment<br>
            Labels: Positive • Negative • Neutral
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("### Live Sentiment Test on Real-World Headlines")

# Example financial headlines
test_headlines = [
    "Bitcoin surges past $69,000 as institutional adoption accelerates",
    "SEC delays ETF decision – market reacts with heavy selling",
    "MicroStrategy adds another 5,000 BTC to corporate treasury",
    "China intensifies crackdown on crypto mining operations",
    "BlackRock files for spot Bitcoin ETF – major bullish catalyst",
    "Mt. Gox begins creditor repayments – $9B BTC overhang looms",
    "Elon Musk tweets support for Dogecoin, BTC holds steady",
    "Federal Reserve signals rate cuts – risk-on environment returns"
]

if st.button("Run Live FinBERT Sentiment Analysis", type="primary", use_container_width=True):
    if not SENTIMENT_AVAILABLE:
        st.error("FinBERT model not available – check src/ml/sentiment.py")
    else:
        with st.spinner("Running FinBERT on 8 financial headlines..."):
            try:
                avg_score, detailed = get_sentiment(test_headlines)

                results_df = pd.DataFrame({
                    "Headline": test_headlines,
                    "Sentiment": [d["label"].title() for d in detailed],
                    "Confidence": [f"{d['score']:.1%}" for d in detailed]
                })

                # Color coding
                def color_sentiment(val):
                    color = "#00ff88" if val == "Positive" else "#ff3366" if val == "Negative" else "#ffaa00"
                    return f'background-color: {color}20; color: {color}; font-weight: bold'

                styled_df = results_df.style.applymap(color_sentiment, subset=["Sentiment"])
                st.dataframe(styled_df, use_container_width=True, height=400)

                # Average sentiment
                pos = sum(1 for d in detailed if d["label"] == "positive")
                neg = sum(1 for d in detailed if d["label"] == "negative")
                neu = len(detailed) - pos - neg

                colA, colB, colC, colD = st.columns(4)
                colA.metric("Positive Headlines", pos, delta=None)
                colB.metric("Negative Headlines", neg, delta=None)
                colC.metric("Neutral Headlines", neu, delta=None)
                colD.metric("Overall Sentiment Score", f"{avg_score:+.4f}", delta=f"{avg_score:+.1%}")

                if avg_score > 0.15:
                    st.success("OVERALL MARKET SENTIMENT: Strongly Bullish")
                elif avg_score > 0.05:
                    st.info("OVERALL MARKET SENTIMENT: Mildly Bullish")
                elif avg_score < -0.15:
                    st.error("OVERALL MARKET SENTIMENT: Strongly Bearish")
                else:
                    st.warning("OVERALL MARKET SENTIMENT: Neutral / Mixed")

            except Exception as e:
                st.error(f"Analysis failed: {e}")

# ====================================================
# SECTION 2: LSTM PRICE MODEL ARCHITECTURE
# ====================================================
st.markdown("---")
st.markdown("<h2 style='text-align: center; color: #00D4FF;'>Price Prediction Engine – Stacked LSTM</h2>", unsafe_allow_html=True)

col_l, col_r = st.columns(2)

with col_l:
    st.markdown("""
    ### Model Architecture
    ```text
    Input Shape          → (60, 6)     [60 hours × 6 features]
    ├─ LSTM Layer 1      → 128 units + Dropout(0.3)
    ├─ LSTM Layer 2      → 64 units  + Dropout(0.3)
    ├─ LSTM Layer 3      → 32 units
    ├─ Dense Layer       → 25 units
    └─ Output Layer      → 1 unit     → Next Hour Close Price Prediction 
    ````
    - **Lookback Window**: 60 hours of historical data 
    - **Features**: OHLCV + Sentiment Score
    - **Optimizer**: Adam
    - **Loss Function**: Mean Squared Error (MSE)
    """)