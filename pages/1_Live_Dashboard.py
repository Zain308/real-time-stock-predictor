import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
import requests
import joblib
import time
from streamlit_autorefresh import st_autorefresh

# --- Safe imports with fallbacks ---
try:
    from src.db_logger import log_prediction_to_db
except ImportError:
    def log_prediction_to_db(df, pred): 
        """Fallback logging"""
        pass

try:
    from src.data_ingestion.news_fetcher import fetch_company_news
except ImportError:
    def fetch_company_news(query): 
        return []  # Fallback empty headlines

try:
    from src.ml.sentiment import get_sentiment
except ImportError:
    def get_sentiment(headlines): 
        return 0.0, []  # Fallback neutral

from src.ml.preprocessing import TIMESTEPS, FEATURES

# ---------------------------------------------------------
# 1. PAGE CONFIG & AUTO-REFRESH
# ---------------------------------------------------------
st.set_page_config(layout="wide", page_title="Live Crypto Dashboard")
st_autorefresh(interval=60 * 1000, key="data_refresher")

st.title("⚡ Real-Time BTC/USD Prediction Engine")
st.caption("Data Source: Kraken Public API (Direct Exchange Feed)")

# ---------------------------------------------------------
# 2. FIXED KRAKEN DATA LOADER (CACHED + HASHABLE)
# ---------------------------------------------------------
@st.cache_data(ttl=55, show_spinner=False)  # Safe cache: refreshes ~every minute
def fetch_kraken_data():
    """
    Fetches last ~120 minutes of BTC/USD 1m candles from Kraken.
    Always returns a clean DataFrame (hashable for caching).
    """
    url = "https://api.kraken.com/0/public/OHLC"
    params = {
        "pair": "XBTUSD",
        "interval": 1  # 1-minute candles
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data.get('error'):
            st.error(f"Kraken API Error: {data['error']}")
            return pd.DataFrame()

        # Kraken's result is a dict with dynamic pair key (e.g., "XXBTZUSD")
        result = data['result']
        if not result:
            return pd.DataFrame()
        
        # Get the pair key (ignore 'last')
        pair_key = next((k for k in result if k != 'last'), None)
        if not pair_key:
            return pd.DataFrame()
        
        ohlc_list = result[pair_key]  # This is the list of [time, o, h, l, c, vwap, vol, count]
        
        if not ohlc_list:
            return pd.DataFrame()

        # FIXED: Convert list-of-lists to DataFrame with proper columns
        df = pd.DataFrame(
            ohlc_list,
            columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count']
        )
        
        # Clean and type-convert (essential for hashing/caching)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop any invalid rows and sort
        df = df[numeric_cols].dropna().sort_index()
        
        # Limit to recent data (e.g., last 120 mins) if too much
        now = pd.Timestamp.now()
        df = df[df.index >= (now - pd.Timedelta(minutes=120))]
        
        return df

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
        st.success("Models loaded successfully!")
    except Exception as e:
        st.error(f"Models not found. Error: {e}")
    return model, scaler

model, scaler = load_models()

# ---------------------------------------------------------
# 4. DASHBOARD VISUALIZATION
# ---------------------------------------------------------
# Load data (cached, so fast after first call)
df = fetch_kraken_data()

col1, col2 = st.columns([1, 2])

with col1:
    if not df.empty:
        # Show recent 60 candles for chart
        chart_data = df.tail(60)
        
        fig = go.Figure(data=[go.Candlestick(
            x=chart_data.index,
            open=chart_data['open'], 
            high=chart_data['high'],
            low=chart_data['low'], 
            close=chart_data['close'],
            increasing_line_color='#00C805', 
            decreasing_line_color='#FF3B30'
        )])
        
        fig.update_layout(
            title='Live Price Action (1m Interval)',
            yaxis_title='Price (USD)',
            template="plotly_dark",
            height=500,
            xaxis_rangeslider_visible=False,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig, width='stretch')  # Fixed deprecation
        st.info(f"Loaded {len(df)} candles | Last: ${chart_data['close'].iloc[-1]:,.2f}")
    else:
        st.warning("Unable to load market data. Retrying in 60s...")

with col2:
    st.subheader("🔮 AI Forecast")
    
    if df.empty:
        st.info("Loading data...")
    elif len(df) < TIMESTEPS:
        st.warning(f"Gathering data... ({len(df)}/{TIMESTEPS} candles needed)")
    else:
        if st.button("Predict Next Close", type="primary", use_container_width=True):
            if model is None or scaler is None:
                st.error("Models missing. Check /models folder.")
            else:
                with st.spinner("Analyzing market patterns + sentiment..."):
                    try:
                        # 1. Sentiment (with fallback)
                        sentiment_score = 0.0
                        try:
                            headlines = fetch_company_news("CRYPTO")
                            if headlines:
                                sentiment_score, _ = get_sentiment(headlines)
                        except Exception as sent_e:
                            st.warning(f"Sentiment unavailable: {sent_e}")

                        st.metric("News Sentiment", f"{sentiment_score:.4f}", 
                                help="From -1 (bearish) to +1 (bullish)")

                        # 2. Prepare input data
                        input_df = df.tail(TIMESTEPS).copy()
                        input_df = input_df[['open', 'high', 'low', 'close', 'volume']]
                        input_df['sentiment'] = sentiment_score  # Add sentiment column

                        # 3. Scale and predict
                        scaled_data = scaler.transform(input_df)
                        model_input = scaled_data.reshape((1, TIMESTEPS, FEATURES))
                        prediction_scaled = model.predict(model_input, verbose=0)[0][0]

                        # 4. FIXED Inverse Transform (dummy array for proper scaling)
                        dummy_row = np.zeros((1, FEATURES))
                        dummy_row[0, 3] = prediction_scaled  # Index 3 = 'close' column
                        predicted_price = scaler.inverse_transform(dummy_row)[0, 3]

                        # 5. Display results
                        current_price = df['close'].iloc[-1]
                        diff = predicted_price - current_price
                        diff_pct = (diff / current_price) * 100
                        
                        st.metric(
                            label="Predicted Price (+1hr)", 
                            value=f"${predicted_price:,.2f}", 
                            delta=f"{diff:+.2f} ({diff_pct:+.1f}%)"
                        )
                        
                        # 6. Log prediction
                        try:
                            log_prediction_to_db(input_df, predicted_price)
                            st.success("Prediction logged to DB!")
                        except Exception as log_e:
                            st.info(f"Logging skipped: {log_e}")

                    except Exception as e:
                        st.error(f"Prediction Error: {e}")
                        st.info("Check models and data length.")

# ---------------------------------------------------------
# 5. RAW DATA EXPANDER
# ---------------------------------------------------------
with st.expander("🔍 View Real-Time Data Feed"):
    if not df.empty:
        st.dataframe(
            df.tail(10).sort_index(ascending=False), 
            use_container_width=False,  # Fixed to width='stretch' equivalent
            width='stretch'
        )
    else:
        st.info("No data available yet.")