import streamlit as st
import time
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import plotly.graph_objects as go
import finnhub
import os
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

# Automatically refresh the page every 60 seconds (60000ms)
st_autorefresh(interval=60 * 1000, key="data_refresher")

# ---------------------------
# 2. INSTANT DATA LOADER (The Fix)
# ---------------------------
@st.cache_data(ttl=60) # Cache data for 60 seconds so we don't spam the API
def load_live_data(symbol="BINANCE:BTCUSDT"):
    """
    Fetches the last 120 minutes of data INSTANTLY from Finnhub.
    No waiting for a buffer to fill.
    """
    try:
        # 1. Get API Key
        api_key = st.secrets.get("FINNHUB_API_KEY")
        if not api_key:
            api_key = os.environ.get("FINNHUB_API_KEY")
        
        if not api_key:
            st.error("API Key not found. Please check secrets.toml")
            return pd.DataFrame()

        # 2. Initialize Client
        finnhub_client = finnhub.Client(api_key=api_key)
        
        # 3. Calculate Time Window (Last 2 hours)
        end_t = int(time.time())
        start_t = end_t - (60 * 120) # 120 minutes ago

        # 4. Fetch Data
        # '1' = 1 minute resolution
        res = finnhub_client.crypto_candles(symbol, '1', start_t, end_t)
        
        # 5. Validate Response
        if res.get('s')!= 'ok':
            st.warning("Finnhub API returned no data (market might be quiet).")
            return pd.DataFrame()
        
        if 't' not in res or not res['t']:
            return pd.DataFrame()

        # 6. Build DataFrame
        df = pd.DataFrame({
            'time': [pd.to_datetime(t, unit='s') for t in res['t']],
            'open': res['o'],
            'high': res['h'],
            'low': res['l'],
            'close': res['c'],
            'volume': res['v']
        })
        df.set_index('time', inplace=True)
        
        # Clean duplicates
        df = df[~df.index.duplicated(keep='last')].sort_index()
        
        return df

    except Exception as e:
        st.error(f"Data Loading Error: {e}")
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
        st.error(f"Error loading models. Run 'python src/ml/prediction.py' locally first. Error: {e}")
    return model, scaler

model, scaler = load_models()

# ---------------------------
# 4. DASHBOARD UI
# ---------------------------
st.title("⚡ Live BTC/USDT Prediction Engine")
st.markdown("Fetching last 2 hours of market data...")

# Load Data Immediately
df = load_live_data()

col1, col2 = st.columns([1, 2])

with col1:
    if not df.empty:
        # Draw Candlestick Chart
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['open'], high=df['high'],
            low=df['low'], close=df['close'],
            increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
        )])
        fig.update_layout(
            title='Real-Time Price Action (1m Candles)',
            yaxis_title='Price (USDT)',
            template="plotly_dark",
            height=500,
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Waiting for API data...")

with col2:
    st.subheader("🤖 AI Prediction")
    
    if df.empty:
        st.warning("Loading data...")
    elif len(df) < TIMESTEPS:
        st.warning(f"Need {TIMESTEPS} candles. Have {len(df)}.")
    else:
        predict_btn = st.button("Predict Next Hour", type="primary")
        
        if predict_btn:
            if model is None or scaler is None:
                st.error("Models missing.")
            else:
                with st.spinner("Analyzing market sentiment & price action..."):
                    try:
                        # A. GET SENTIMENT
                        try:
                            headlines = fetch_company_news("CRYPTO")
                            sentiment_score, _ = get_sentiment(headlines)
                        except:
                            sentiment_score = 0.0 # Neutral fallback
                        
                        st.metric("Live News Sentiment", f"{sentiment_score:.4f}")

                        # B. PREPARE DATA
                        # Get exactly the last 60 candles
                        input_df = df.tail(TIMESTEPS).copy()
                        input_df = input_df[['open', 'high', 'low', 'close', 'volume']]
                        input_df['sentiment'] = sentiment_score

                        # C. SCALE & RESHAPE
                        scaled = scaler.transform(input_df)
                        # Reshape to (1, 60, 6)
                        model_input = scaled.reshape((1, TIMESTEPS, FEATURES))

                        # D. PREDICT
                        prediction_scaled = model.predict(model_input)
                        pred_value_scaled = float(prediction_scaled)

                        # E. INVERSE TRANSFORM
                        # We need a dummy array of shape (1, 6) to reverse the scaler
                        dummy = np.zeros((1, FEATURES))
                        dummy = pred_value_scaled # Index 3 is 'close'
                        
                        real_price = scaler.inverse_transform(dummy)
                        
                        # F. DISPLAY
                        current_price = float(df['close'].iloc[-1])
                        diff = real_price - current_price
                        
                        st.metric(
                            label="Predicted Close (Next Hour)",
                            value=f"${real_price:,.2f}",
                            delta=f"{diff:+.2f}"
                        )

                        # G. LOGGING
                        log_prediction_to_db(input_df, real_price)

                    except Exception as e:
                        st.error(f"Prediction Failed: {e}")

# Display Raw Data
with st.expander("View Raw Data"):
    st.dataframe(df.sort_index(ascending=False), use_container_width=True)