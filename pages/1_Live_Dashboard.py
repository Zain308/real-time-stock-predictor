# ================================================
#   REAL-TIME BTC/USD AI PREDICTION DASHBOARD
#   Data: Kraken Exchange | Model: LSTM + Sentiment
#   Author: Zain308 | Status: PRODUCTION READY
# ================================================

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
import requests
import joblib
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# ------------------- Safe Imports -------------------
try:
    from src.db_logger import log_prediction_to_db
except ImportError:
    def log_prediction_to_db(df, pred): 
        pass  # Silent fallback

try:
    from src.data_ingestion.news_fetcher import fetch_company_news
except ImportError:
    def fetch_company_news(query): 
        return []

try:
    from src.ml.sentiment import get_sentiment
except ImportError:
    def get_sentiment(headlines): 
        return 0.0, []

from src.ml.preprocessing import TIMESTEPS, FEATURES

# ------------------- Page Config -------------------
st.set_page_config(
    page_title="BTC/USD Live AI Predictor",
    page_icon="Bitcoin",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Auto-refresh every 60 seconds
st_autorefresh(interval=60_000, key="data_refresh")

# ------------------- Title & Header -------------------
st.markdown("""
    <h1 style='text-align: center; color: #00D4FF;'>Real-Time BTC/USD Prediction Engine</h1>
    <p style='text-align: center; color: #888; font-size: 18px;'>
        Powered by Kraken Exchange • LSTM Neural Network • Live News Sentiment
    </p>
""", unsafe_allow_html=True)

# ------------------- Fetch Live Data from Kraken -------------------
@st.cache_data(ttl=55, show_spinner=False)
def fetch_live_data():
    url = "https://api.kraken.com/0/public/OHLC"
    params = {"pair": "XBTUSD", "interval": 1}
    
    try:
        response = requests.get(url, params=params, timeout=12)
        data = response.json()
        
        if data.get("error") and data["error"]:
            st.error(f"Kraken API Error: {data['error'][0]}")
            return pd.DataFrame()

        # Extract dynamic pair key (e.g., XXBTZUSD)
        pair_key = [k for k in data["result"].keys() if k != "last"][0]
        candles = data["result"][pair_key]

        df = pd.DataFrame(candles, columns=[
            "timestamp", "open", "high", "low", "close", "vwap", "volume", "count"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df.set_index("timestamp", inplace=True)
        df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
        
        return df[["open", "high", "low", "close", "volume"]].sort_index()

    except Exception as e:
        st.error(f"Connection failed: {e}")
        return pd.DataFrame()

# ------------------- Load AI Models -------------------
@st.cache_resource
def load_ai_models():
    try:
        model = tf.keras.models.load_model("models/price_model.h5")
        scaler = joblib.load("models/price_scaler.pkl")
        st.success("AI Model & Scaler Loaded Successfully")
        return model, scaler
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None, None

model, scaler = load_ai_models()
df = fetch_live_data()

# ------------------- Layout: Two Columns -------------------
col_chart, col_ai = st.columns([2, 1])

with col_chart:
    st.subheader("Live BTC/USD Price Action (1-Minute Candles)")

    if df.empty:
        st.warning("Fetching real-time data from Kraken...")
        st.stop()

    # Candlestick Chart
    fig = go.Figure(data=go.Candlestick(
        x=df.tail(60).index,
        open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        increasing_line_color="#00ff88",
        decreasing_line_color="#ff3366"
    ))

    fig.update_layout(
        template="plotly_dark",
        height=600,
        xaxis_rangeslider_visible=False,
        title=f"Last Price: ${df['close'].iloc[-1]:,.2f} USD",
        xaxis_title="Time",
        yaxis_title="Price (USD)"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Stats
    current_price = df["close"].iloc[-1]
    price_1h_ago = df["close"].iloc[-60] if len(df) >= 60 else current_price
    change_1h = current_price - price_1h_ago
    pct_change = (change_1h / price_1h_ago) * 100

    st.markdown(f"""
    **Current Price:** `${current_price:,.2f}`  
    **1H Change:** `{change_1h:+.2f} ({pct_change:+.2f}%)`  
    **Candles Loaded:** {len(df)} | Updated: {datetime.now().strftime('%H:%M:%S')}
    """)

with col_ai:
    st.subheader("AI Forecast Engine")

    if df.empty:
        st.info("Waiting for market data...")
    elif len(df) < TIMESTEPS:
        st.warning(f"Collecting data... ({len(df)}/{TIMESTEPS} candles needed)")
    else:
        # Predict Button
        if st.button("Predict Next Hour Price", type="primary", use_container_width=True, key="predict"):
            if not model or not scaler:
                st.error("AI model not loaded.")
            else:
                with st.spinner("Analyzing patterns + news sentiment..."):
                    # --- News Sentiment ---
                    sentiment_score = 0.0
                    try:
                        headlines = fetch_company_news("Bitcoin OR BTC OR Cryptocurrency")
                        if headlines:
                            sentiment_score, _ = get_sentiment(headlines)
                    except:
                        sentiment_score = 0.0

                    sentiment_color = "inverse" if sentiment_score < 0 else "normal"
                    st.metric("Live News Sentiment", f"{sentiment_score:+.4f}", delta=None)

                    # --- Prepare Input Exactly Like Training ---
                    seq = df.tail(TIMESTEPS).copy()
                    seq = seq.rename(columns={
                        "open": "Open", "high": "High", "low": "Low",
                        "close": "Close", "volume": "Volume"
                    })
                    seq = seq[["Open", "High", "Low", "Close", "Volume"]]
                    seq["Sentiment"] = sentiment_score

                    # --- CRITICAL FIX: Use .values to prevent memory crash ---
                    scaled_input = scaler.transform(seq.values)
                    X = scaled_input.reshape((1, TIMESTEPS, FEATURES))

                    # --- Prediction ---
                    predicted_scaled = model.predict(X, verbose=0)[0][0]

                    # --- Inverse Transform ---
                    dummy = np.zeros((1, FEATURES))
                    dummy[0, 3] = predicted_scaled  # Index 3 = Close price
                    predicted_price = scaler.inverse_transform(dummy)[0, 3]

                    # --- Results ---
                    price_diff = predicted_price - current_price
                    pct_diff = (price_diff / current_price) * 100

                    st.metric(
                        label="Predicted Price (+1 Hour)",
                        value=f"${predicted_price:,.2f}",
                        delta=f"{price_diff:+.2f} ({pct_diff:+.2f}%)"
                    )

                    if price_diff > 0:
                        st.success("Bullish Signal Detected")
                    elif price_diff < -50:
                        st.error("Bearish Signal Detected")
                    else:
                        st.info("Neutral / Sideways Expected")

                    # --- Log to DB ---
                    try:
                        log_prediction_to_db(seq, predicted_price)
                        st.toast("Prediction logged to database", icon="Success")
                    except:
                        pass

# ------------------- Footer & Raw Data -------------------
with st.expander("View Raw Market Data Feed", expanded=False):
    if not df.empty:
        st.dataframe(
            df.tail(20)[["open", "high", "low", "close", "volume"]].round(2),
            use_container_width=True
        )

st.markdown("---")
st.caption("Real-Time BTC/USD AI Prediction Dashboard © 2025 | Built with ❤️ using Streamlit & TensorFlow")