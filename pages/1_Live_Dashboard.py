import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logs
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
import requests
import joblib
import time
import warnings
from streamlit_autorefresh import st_autorefresh

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Safe imports
try:
    from src.db_logger import log_prediction_to_db
except:
    def log_prediction_to_db(*args): pass

try:
    from src.data_ingestion.news_fetcher import fetch_company_news
except:
    def fetch_company_news(q): return []

try:
    from src.ml.sentiment import get_sentiment
except:
    def get_sentiment(h): return 0.0, []

from src.ml.preprocessing import TIMESTEPS, FEATURES

# ====================== CONFIG ======================
st.set_page_config(layout="wide", page_title="BTC Live Prediction")
st_autorefresh(interval=60_000, key="refresh")

st.title("Real-Time BTC/USD Prediction Engine")
st.caption("Data: Kraken Public API • 1-Minute Candles • LSTM + Sentiment")

# ====================== KRAKEN DATA ======================
@st.cache_data(ttl=55, show_spinner=False)
def get_kraken_data():
    try:
        url = "https://api.kraken.com/0/public/OHLC"
        params = {"pair": "XBTUSD", "interval": 1}
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if data["error"]:
            return pd.DataFrame()

        key = [k for k in data["result"].keys() if k != "last"][0]
        df = pd.DataFrame(data["result"][key], columns=[
            "time", "open", "high", "low", "close", "vwap", "volume", "count"
        ])
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        df = df.astype(float)
        return df[["open", "high", "low", "close", "volume"]].sort_index()
    except:
        return pd.DataFrame()

# ====================== LOAD MODELS ======================
@st.cache_resource
def load_models():
    try:
        model = tf.keras.models.load_model("models/price_model.h5")
        model.compile(optimizer='adam', loss='mse')  # Suppress metrics warning
        scaler = joblib.load("models/price_scaler.pkl")
        st.success("Models loaded successfully")
        return model, scaler
    except Exception as e:
        st.error(f"Model load failed: {e}")
        return None, None

model, scaler = load_models()

# ====================== FETCH DATA ======================
df = get_kraken_data()

col1, col2 = st.columns([1, 2])

# ====================== CHART ======================
with col1:
    if not df.empty:
        chart = df.tail(60)
        fig = go.Figure(go.Candlestick(
            x=chart.index,
            open=chart["open"], high=chart["high"],
            low=chart["low"], close=chart["close"],
            increasing_line_color="#00ff88", decreasing_line_color="#ff3366"
        ))
        fig.update_layout(
            title="Live BTC/USD (1m)",
            template="plotly_dark",
            height=520,
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Last price: ${df['close'].iloc[-1]:,.2f} • Candles: {len(df)}")
    else:
        st.warning("Loading live data...")

# ====================== PREDICTION ======================
with col2:
    st.subheader("AI Forecast")

    if df.empty:
        st.info("Waiting for market data...")
    elif len(df) < TIMESTEPS:
        st.warning(f"Need {TIMESTEPS} candles • Have {len(df)}")
    else:
        if st.button("Predict Next Hour Price", type="primary", use_container_width=True):
            if not model or not scaler:
                st.error("Models not loaded.")
            else:
                with st.spinner("Predicting..."):
                    # Sentiment
                    sentiment = 0.0
                    try:
                        headlines = fetch_company_news("CRYPTO")
                        if headlines:
                            sentiment, _ = get_sentiment(headlines)
                    except:
                        pass
                    st.metric("News Sentiment", f"{sentiment:+.4f}")

                    # Prepare exact feature names
                    seq = df.tail(TIMESTEPS).copy()
                    seq = seq.rename(columns={
                        "open": "Open", "high": "High", "low": "Low",
                        "close": "Close", "volume": "Volume"
                    })
                    seq = seq[["Open", "High", "Low", "Close", "Volume"]]
                    seq["Sentiment"] = sentiment

                    # Transform using DataFrame
                    scaled = scaler.transform(seq)
                    X = scaled.reshape(1, TIMESTEPS, FEATURES)

                    # Predict
                    pred_scaled = model.predict(X, verbose=0)[0][0]

                    # Inverse transform
                    dummy = np.zeros((1, FEATURES))
                    dummy[0, 3] = pred_scaled
                    predicted = scaler.inverse_transform(dummy)[0, 3]

                    current = df["close"].iloc[-1]
                    change = predicted - current
                    pct = change / current * 100

                    st.metric(
                        "Predicted Price (+1h)",
                        f"${predicted:,.2f}",
                        f"{change:+.2f} ({pct:+.2f}%)"
                    )

                    try:
                        log_prediction_to_db(seq, predicted)
                        st.success("Logged to DB")
                    except:
                        pass

# ====================== RAW DATA ======================
with st.expander("View Live Feed"):
    if not df.empty:
        st.dataframe(df.tail(10).sort_index(ascending=False), use_container_width=True)