import streamlit as st
import time
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import plotly.graph_objects as go
import requests
from streamlit_autorefresh import st_autorefresh

# Import our custom modules
from src.logging import log_prediction_to_db
from src.data_ingestion.news_fetcher import fetch_company_news
from src.ml.sentiment import get_sentiment
from src.ml.preprocessing import TIMESTEPS, FEATURES

# ---------------------------
# 1. CONFIGURATION & AUTO-REFRESH
# ---------------------------
st.set_page_config(layout="wide", page_title="Live Crypto Dashboard")

# Refresh the page every 60 seconds
st_autorefresh(interval=60 * 1000, key="data_refresher")

# ---------------------------
# 2. ROBUST KRAKEN DATA LOADER (No API Key Needed)
# ---------------------------
@st.cache_data(ttl=60)
def load_live_data():
    """
    Fetches the last 720 minutes of BTC/USD data directly from Kraken Exchange.
    This API is public, free, and stable on Streamlit Cloud.
    """
    url = "https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=1"
    
    try:
        # 1. Fetch Data
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        data = response.json()
        
        if data.get('error'):
            st.error(f"Kraken API Error: {data['error']}")
            return pd.DataFrame()

        # 2. Parse Data (Kraken returns data under key 'XXBTZUSD')
        # Format: [time, open, high, low, close, vwap, volume, count]
        candles = data['result']
        
        df = pd.DataFrame(candles, columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
        
        # 3. Convert Types
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
            
        # 4. Sort and return
        return df.sort_index()

    except Exception as e:
        st.error(f"Data Loading Failed: {e}")
        return pd.DataFrame()

# ---------------------------
# 3. MODEL LOADER
# ---------------------------
@st.cache_resource
def load_models():
    model, scaler = None, None
    try:
        model = tf.keras.models.load_model('models/price_model.h5')
        scaler = joblib.load('models/price_scaler.pkl')
    except Exception as e:
        st.error(f"Error loading models. Please run 'python -m src.ml.prediction' locally first.")
    return model, scaler

model, scaler = load_models()

# ---------------------------
# 4. DASHBOARD UI
# ---------------------------
st.title("⚡ Live BTC/USD Prediction Engine")
st.caption("Data Source: Kraken Public API (Direct Exchange Feed)")

# Load Data Immediately
df = load_live_data()

col1, col2 = st.columns([1, 2])

with col1:
    if not df.empty:
        # Get last 60 candles for the chart context
        chart_data = df.tail(120)
        
        fig = go.Figure(data=[go.Candlestick(
            x=chart_data.index,
            open=chart_data['open'], high=chart_data['high'],
            low=chart_data['low'], close=chart_data['close'],
            increasing_line_color='#00ff00', decreasing_line_color='#ff0000'
        )])
        fig.update_layout(
            title='Real-Time Price Action (1m Candles)',
            yaxis_title='Price (USD)',
            template="plotly_dark",
            height=500,
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Connecting to Kraken Exchange...")

with col2:
    st.subheader("🤖 AI Prediction")
    
    if df.empty:
        st.warning("Waiting for data...")
    else:
        predict_btn = st.button("Predict Next Hour", type="primary")
        
        if predict_btn:
            if model is None or scaler is None:
                st.error("Models missing.")
            else:
                with st.spinner("Processing..."):
                    try:
                        # A. SAFE SENTIMENT FETCHING
                        try:
                            # Finnhub free tier often fails for crypto news. 
                            # We wrap this to prevent the whole app from crashing.
                            headlines = fetch_company_news("CRYPTO")
                            if headlines:
                                sentiment_score, _ = get_sentiment(headlines)
                            else:
                                sentiment_score = 0.0
                        except:
                            sentiment_score = 0.0 # Default to neutral if API fails
                        
                        st.metric("Market Sentiment", f"{sentiment_score:.4f}")

                        # B. PREPARE DATA (Last 60 points)
                        input_df = df.tail(TIMESTEPS).copy()
                        input_df = input_df[['open', 'high', 'low', 'close', 'volume']]
                        input_df['sentiment'] = sentiment_score

                        # C. RUN PREDICTION PIPELINE
                        scaled = scaler.transform(input_df)
                        model_input = scaled.reshape((1, TIMESTEPS, FEATURES))
                        
                        prediction_scaled = model.predict(model_input)
                        pred_value_scaled = float(prediction_scaled)

                        # D. INVERSE TRANSFORM
                        dummy = np.zeros((1, FEATURES))
                        dummy = pred_value_scaled # Index 3 is 'close'
                        real_price = scaler.inverse_transform(dummy)
                        
                        # E. DISPLAY RESULT
                        current_price = float(df['close'].iloc[-1])
                        diff = real_price - current_price
                        
                        st.metric(
                            label="Predicted Close",
                            value=f"${real_price:,.2f}",
                            delta=f"{diff:+.2f}"
                        )
                        
                        # Log success
                        try:
                            log_prediction_to_db(input_df, real_price)
                        except:
                            pass

                    except Exception as e:
                        st.error(f"Prediction Error: {e}")

# Display Raw Data
with st.expander("View Live Data Feed"):
    if not df.empty:
        st.dataframe(df.tail(10).sort_index(ascending=False), use_container_width=True)