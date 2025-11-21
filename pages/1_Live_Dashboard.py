import streamlit as st
import time
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import plotly.graph_objects as go
import finnhub
import os
from streamlit_autorefresh import st_autorefresh # type: ignore

# --- FIX: Import from the NEW db_logger file ---
from src.db_logger import log_prediction_to_db

from src.data_ingestion.news_fetcher import fetch_company_news
from src.ml.sentiment import get_sentiment
from src.ml.preprocessing import TIMESTEPS, FEATURES

# ---------------------------
# 1. CONFIGURATION & AUTO-REFRESH
# ---------------------------
st.set_page_config(layout="wide", page_title="Live Crypto Dashboard")
st_autorefresh(interval=60 * 1000, key="data_refresher")

# ---------------------------
# 2. INSTANT DATA LOADER
# ---------------------------
@st.cache_data(ttl=60)
def load_live_data(symbol="BINANCE:BTCUSDT"):
    try:
        # Get API Key
        api_key = st.secrets.get("FINNHUB_API_KEY")
        if not api_key:
            api_key = os.environ.get("FINNHUB_API_KEY")
        
        if not api_key:
            st.error("API Key not found. Please add FINNHUB_API_KEY to Streamlit Secrets.")
            return pd.DataFrame()

        finnhub_client = finnhub.Client(api_key=api_key)
        
        # Calculate Time Window (Last 2 hours)
        to_ts = int(time.time())
        from_ts = to_ts - (60 * 120)

        res = finnhub_client.crypto_candles(symbol, '1', from_ts, to_ts)
        
        if res.get('s')!= 'ok':
            return pd.DataFrame()

        df = pd.DataFrame({
            'time': [pd.to_datetime(t, unit='s') for t in res['t']],
            'open': res['o'],
            'high': res['h'],
            'low': res['l'],
            'close': res['c'],
            'volume': res['v']
        })
        df.set_index('time', inplace=True)
        return df.sort_index()

    except Exception as e:
        st.error(f"Data Error: {e}")
        return pd.DataFrame()

# ---------------------------
# 3. LOAD MODELS
# ---------------------------
@st.cache_resource
def load_models():
    model, scaler = None, None
    try:
        model = tf.keras.models.load_model('models/price_model.h5')
        scaler = joblib.load('models/price_scaler.pkl')
    except Exception as e:
        st.error(f"Error loading models: {e}")
    return model, scaler

model, scaler = load_models()

# ---------------------------
# 4. DASHBOARD UI
# ---------------------------
st.title("⚡ Live BTC/USDT Prediction Engine")
st.caption("Data Source: Finnhub (Updates every 60s)")

df = load_live_data()

col1, col2 = st.columns([1, 2])

with col1:
    if not df.empty:
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['open'], high=df['high'],
            low=df['low'], close=df['close'],
            increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
        )])
        fig.update_layout(
            title='Real-Time Price Action',
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
        st.warning("Loading...")
    elif len(df) < TIMESTEPS:
        st.warning(f"Need {TIMESTEPS} candles. Have {len(df)}.")
    else:
        if st.button("Predict Next Hour", type="primary"):
            if model and scaler:
                with st.spinner("Analyzing..."):
                    try:
                        # Sentiment
                        try:
                            headlines = fetch_company_news("CRYPTO")
                            sentiment, _ = get_sentiment(headlines)
                        except:
                            sentiment = 0.0
                        
                        st.metric("Sentiment Score", f"{sentiment:.4f}")

                        # Prepare Data
                        input_df = df.tail(TIMESTEPS).copy()
                        input_df = input_df[['open', 'high', 'low', 'close', 'volume']]
                        input_df['sentiment'] = sentiment

                        # Predict
                        scaled = scaler.transform(input_df)
                        model_input = scaled.reshape((1, TIMESTEPS, FEATURES))
                        prediction = model.predict(model_input)
                        
                        # Inverse Transform
                        dummy = np.zeros((1, FEATURES))
                        dummy = float(prediction) # 3 is close index
                        real_price = scaler.inverse_transform(dummy)
                        
                        curr = float(df['close'].iloc[-1])
                        diff = real_price - curr
                        
                        st.metric("Predicted Close", f"${real_price:,.2f}", f"{diff:+.2f}")

                        # Log
                        try:
                            log_prediction_to_db(input_df, real_price)
                        except:
                            pass
                    except Exception as e:
                        st.error(f"Error: {e}")

with st.expander("View Raw Data"):
    st.dataframe(df.sort_index(ascending=False), use_container_width=True)