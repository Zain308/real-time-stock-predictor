import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
import requests
import joblib
import time
from streamlit_autorefresh import st_autorefresh

# --- FIX: Updated import to match the new filename ---
try:
    from src.db_logger import log_prediction_to_db
except ImportError:
    # Fallback safety if file isn't renamed yet
    pass

from src.data_ingestion.news_fetcher import fetch_company_news
from src.ml.sentiment import get_sentiment
from src.ml.preprocessing import TIMESTEPS, FEATURES

# ---------------------------------------------------------
# 1. PAGE CONFIG & AUTO-REFRESH
# ---------------------------------------------------------
st.set_page_config(layout="wide", page_title="Live Crypto Dashboard")

# Refresh the page every 60 seconds to get the latest candle
st_autorefresh(interval=60 * 1000, key="data_refresher")

st.title("⚡ Real-Time BTC/USD Prediction Engine")
st.caption("Data Source: Kraken Public API (Direct Exchange Feed)")

# ---------------------------------------------------------
# 2. DATA INGESTION (The Fix: Kraken Public API)
# ---------------------------------------------------------
@st.cache_data(ttl=60) # Cache for 60s to prevent rate limits
def fetch_kraken_data():
    """
    Fetches the last 720 minutes (12 hours) of BTC/USD data from Kraken.
    No API Key required. No 403 Errors.
    """
    # Kraken uses 'XBT' instead of 'BTC'
    url = "https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=1"
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        data = response.json()
        
        if data.get('error'):
            st.error(f"Kraken API Error: {data['error']}")
            return pd.DataFrame()

        # Kraken returns data under the key 'XXBTZUSD'
        # We find the result key dynamically to be safe
        result_data = data['result']
        target_key = [k for k in result_data.keys() if k!= 'last']
        ohlc = result_data[target_key]
        
        # Convert to DataFrame
        # Columns: [time, open, high, low, close, vwap, volume, count]
        df = pd.DataFrame(ohlc, columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
        
        # Clean types
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        cols = ['open', 'high', 'low', 'close', 'volume']
        for c in cols:
            df[c] = df[c].astype(float)
            
        return df.sort_index()

    except Exception as e:
        st.error(f"Data Loading Failed: {e}")
        return pd.DataFrame()

# ---------------------------------------------------------
# 3. MODEL LOADING
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    model, scaler = None, None
    try:
        model = tf.keras.models.load_model('models/price_model.h5')
        scaler = joblib.load('models/price_scaler.pkl')
    except Exception as e:
        st.error("Models not found. Please run 'python -m src.ml.prediction' locally to generate them.")
    return model, scaler

model, scaler = load_models()

# ---------------------------------------------------------
# 4. DASHBOARD VISUALIZATION
# ---------------------------------------------------------
# Load data instantly - NO WAITING
df = fetch_kraken_data()

col1, col2 = st.columns([2, 3])

with col1:
    if not df.empty:
        # Show last 60 candles
        chart_data = df.tail(60)
        
        fig = go.Figure(data=[go.Candlestick(
            x=chart_data.index,
            open=chart_data['open'], high=chart_data['high'],
            low=chart_data['low'], close=chart_data['close'],
            increasing_line_color='#00C805', decreasing_line_color='#FF3B30'
        )])
        
        fig.update_layout(
            title='Live Price Action (1m Interval)',
            yaxis_title='Price (USD)',
            template="plotly_dark",
            height=500,
            xaxis_rangeslider_visible=False,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig, use_container_width=True) 
    else:
        st.warning("Unable to load market data. Retrying...")

with col2:
    st.subheader("🔮 AI Forecast")
    
    if df.empty:
        st.info("Loading...")
    elif len(df) < TIMESTEPS:
        st.warning(f"Gathering data... ({len(df)}/{TIMESTEPS})")
    else:
        if st.button("Predict Next Close", type="primary", use_container_width=True):
            if model is None:
                st.error("Models missing.")
            else:
                with st.spinner("Analyzing market patterns..."):
                    try:
                        # 1. Sentiment (Safe Mode)
                        sentiment_score = 0.0
                        try:
                            headlines = fetch_company_news("CRYPTO")
                            if headlines:
                                sentiment_score, _ = get_sentiment(headlines)
                        except Exception:
                            pass # Fail silently to neutral if news API fails
                        
                        st.metric("News Sentiment", f"{sentiment_score:.4f}", help="-1 (Neg) to +1 (Pos)")

                        # 2. Prepare Data
                        input_df = df.tail(TIMESTEPS).copy()
                        input_df = input_df[['open', 'high', 'low', 'close', 'volume']]
                        input_df = sentiment_score # Must match column name in preprocessing

                        # 3. Predict
                        scaled = scaler.transform(input_df)
                        model_input = scaled.reshape((1, TIMESTEPS, FEATURES))
                        prediction = float(model.predict(model_input))

                        # 4. Inverse Transform
                        dummy = np.zeros((1, FEATURES))
                        dummy = prediction # Index 3 is 'close'
                        real_price = scaler.inverse_transform(dummy)
                        
                        # 5. Display
                        current = df['close'].iloc[-1]
                        diff = real_price - current
                        
                        st.metric(
                            label="Predicted Price (+1hr)", 
                            value=f"${real_price:,.2f}", 
                            delta=f"{diff:+.2f}"
                        )
                        
                        # 6. Log
                        try:
                            log_prediction_to_db(input_df, real_price)
                        except:
                            pass

                    except Exception as e:
                        st.error(f"Prediction Error: {e}")

# Raw Data Expander
with st.expander("🔍 View Real-Time Data Feed"):
    if not df.empty:
        st.dataframe(df.tail(10).sort_index(ascending=False), use_container_width=True)