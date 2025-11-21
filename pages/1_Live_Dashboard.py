import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
import requests
import joblib
import time
from streamlit_autorefresh import st_autorefresh

# Safe imports
try:
    from src.db_logger import log_prediction_to_db
except ImportError:
    def log_prediction_to_db(df, pred): pass

try:
    from src.data_ingestion.news_fetcher import fetch_company_news
except ImportError:
    def fetch_company_news(q): return []

try:
    from src.ml.sentiment import get_sentiment
except ImportError:
    def get_sentiment(h): return 0.0, []

from src.ml.preprocessing import TIMESTEPS, FEATURES

# ====================== PAGE CONFIG ======================
st.set_page_config(layout="wide", page_title="Live BTC/USD Prediction Engine")
st_autorefresh(interval=60 * 1000, key="auto_refresh")

st.title("Real-Time BTC/USD Prediction Engine")
st.caption("Data Source: Kraken Public API (Direct Exchange Feed)")

# ====================== KRAKEN DATA LOADER ======================
@st.cache_data(ttl=60)
def fetch_kraken_data():
    try:
        url = "https://api.kraken.com/0/public/OHLC"
        params = {"pair": "XBTUSD", "interval": 1}
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if data.get("error"):
            st.error(f"Kraken Error: {data['error']}")
            return pd.DataFrame()

        pair_key = [k for k in data["result"].keys() if k != "last"][0]
        candles = data["result"][pair_key]

        df = pd.DataFrame(candles, columns=[
            "time", "open", "high", "low", "close", "vwap", "volume", "count"
        ])
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        df = df.astype(float)
        return df[["open", "high", "low", "close", "volume"]].sort_index()

    except Exception as e:
        st.error(f"Data error: {e}")
        return pd.DataFrame()

# ====================== LOAD MODELS ======================
@st.cache_resource
def load_models():
    try:
        model = tf.keras.models.load_model("models/price_model.h5")
        scaler = joblib.load("models/price_scaler.pkl")
        st.success("Models loaded successfully!")
        return model, scaler
    except Exception as e:
        st.error(f"Model load failed: {e}")
        return None, None

model, scaler = load_models()

# ====================== GET DATA ======================
df = fetch_kraken_data()

# ====================== LAYOUT ======================
col1, col2 = st.columns([1, 2])

with col1:
    if not df.empty:
        chart_data = df.tail(60)
        fig = go.Figure(go.Candlestick(
            x=chart_data.index,
            open=chart_data["open"],
            high=chart_data["high"],
            low=chart_data["low"],
            close=chart_data["close"],
            increasing_line_color="#00ff99",
            decreasing_line_color="#ff3366"
        ))
        fig.update_layout(
            title="Live BTC/USD Price (1m candles)",
            template="plotly_dark",
            height=550,
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"Loaded {len(df)} candles | Current: ${df['close'].iloc[-1]:,.2f}")
    else:
        st.warning("Loading live data from Kraken...")

with col2:
    st.subheader("AI Price Forecast")

    if df.empty:
        st.info("Waiting for market data...")
    elif len(df) < TIMESTEPS:
        st.warning(f"Collecting data... ({len(df)}/{TIMESTEPS} candles)")
    else:
        if st.button("Predict Next Hour Price", type="primary", use_container_width=True):
            if not model or not scaler:
                st.error("Models not loaded!")
            else:
                with st.spinner("Running AI prediction..."):
                    # Sentiment
                    sentiment = 0.0
                    try:
                        headlines = fetch_company_news("CRYPTO")
                        if headlines:
                            sentiment, _ = get_sentiment(headlines)
                    except:
                        pass
                    st.metric("News Sentiment Score", f"{sentiment:+.4f}")

                    # === ONLY FIX: Match your training column names exactly ===
                    seq = df.tail(TIMESTEPS).copy()
                    seq = seq.rename(columns={
                        "open": "Open",
                        "high": "High", 
                        "low": "Low",
                        "close": "Close",
                        "volume": "Volume"
                    })
                    seq = seq[["Open", "High", "Low", "Close", "Volume"]]
                    seq["Sentiment"] = sentiment

                    # Predict
                    scaled = scaler.transform(seq)
                    X = scaled.reshape((1, TIMESTEPS, FEATURES))
                    pred_scaled = model.predict(X, verbose=0)[0][0]

                    # Inverse
                    dummy = np.zeros((1, FEATURES))
                    dummy[0, 3] = pred_scaled
                    predicted_price = scaler.inverse_transform(dummy)[0, 3]

                    current = df["close"].iloc[-1]
                    change = predicted_price - current
                    pct = change / current * 100

                    st.metric(
                        "Predicted Price (1h ahead)",
                        f"${predicted_price:,.2f}",
                        f"{change:+.2f} ({pct:+.2f}%)"
                    )

                    try:
                        log_prediction_to_db(seq, predicted_price)
                        st.success("Prediction logged!")
                    except:
                        pass

# Raw data
with st.expander("View Raw Live Data"):
    if not df.empty:
        st.dataframe(df.tail(20).sort_index(ascending=False), use_container_width=True)