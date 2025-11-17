import streamlit as st
import queue
import threading
import time
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import plotly.graph_objects as go
from streamlit.runtime.scriptrunner import add_script_run_ctx
from src.logging import log_prediction_to_db

# ML + News Imports
from src.data_ingestion.news_fetcher import fetch_company_news
from src.ml.sentiment import get_sentiment
from src.ml.preprocessing import TIMESTEPS, FEATURES

# YFinance Poller
import yfinance as yf


# =====================================================
# PRICE POLLER (REPLACES WEBSOCKET)
# =====================================================
class PricePoller:
    def __init__(self, message_queue, symbol="BTC-USD", poll_interval=15):
        self.queue = message_queue
        self.symbol = symbol
        self.poll_interval = poll_interval
        self._stopped = False

    def stop(self):
        self._stopped = True

    def _fetch_latest_minute(self):
        """Fetches last 5m of 1m candles from YFinance."""
        try:
            df = yf.download(self.symbol, period="5m", interval="1m", progress=False)
            if df.empty:
                return None

            last = df.iloc[-1]
            ts = int(pd.to_datetime(df.index[-1]).timestamp() * 1000)

            return {
                "time": ts,
                "open": float(last["Open"]),
                "high": float(last["High"]),
                "low": float(last["Low"]),
                "close": float(last["Close"]),
                "volume": float(last["Volume"])
            }
        except:
            return None

    def start_polling(self):
        while not self._stopped:
            item = self._fetch_latest_minute()
            if item:
                try:
                    self.queue.put_nowait(item)
                except queue.Full:
                    pass
            time.sleep(self.poll_interval)


# =====================================================
# SESSION STATE INITIALIZATION
# =====================================================
if "message_queue" not in st.session_state:
    st.session_state.message_queue = queue.Queue(maxsize=2000)

if "live_data" not in st.session_state:
    st.session_state.live_data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

if "poller_thread" not in st.session_state:
    poller = PricePoller(st.session_state.message_queue, "BTC-USD", 15)
    t = threading.Thread(target=poller.start_polling, daemon=True)
    add_script_run_ctx(t)
    t.start()
    st.session_state.poller_thread = t
    st.session_state.poller = poller


# =====================================================
# LOAD MODELS
# =====================================================
@st.cache_resource
def load_models():
    try:
        model = tf.keras.models.load_model("models/price_model.h5")
        scaler = joblib.load("models/price_scaler.pkl")
        return model, scaler
    except:
        return None, None

model, scaler = load_models()


# =====================================================
# UI
# =====================================================
st.title("Live BTC/USDT Price & Prediction Dashboard")

col1, col2 = st.columns([1, 2])

with col1:
    chart_placeholder = st.empty()

with col2:
    st.subheader("🔍 On-Demand Prediction")
    predict_button = st.button("Predict Next Hour Price")
    prediction_placeholder = st.empty()

    st.subheader("📊 Live Sentiment (Last 24h)")
    sentiment_placeholder = st.empty()

data_grid_placeholder = st.empty()


# =====================================================
# PREDICTION LOGIC
# =====================================================
if predict_button:
    if (model is None) or (scaler is None):
        prediction_placeholder.error("Model not loaded.")
    else:
        if len(st.session_state.live_data) < TIMESTEPS:
            prediction_placeholder.warning(f"Need {TIMESTEPS} data points.")
        else:
            with st.spinner("Running prediction..."):
                try:
                    # Get Sentiment
                    try:
                        headlines = fetch_company_news("CRYPTO")
                        live_sentiment_score, _ = get_sentiment(headlines)
                    except:
                        live_sentiment_score = 0.0

                    sentiment_placeholder.write(f"Current Sentiment: {live_sentiment_score:.4f}")

                    # Prepare data
                    features_df = st.session_state.live_data.tail(TIMESTEPS)[
                        ['open', 'high', 'low', 'close', 'volume']
                    ].copy()
                    features_df["sentiment"] = live_sentiment_score

                    # Correct Feature Check
                    if features_df.shape[1] != FEATURES:
                        prediction_placeholder.error(
                            f"Feature mismatch: expected {FEATURES}, got {features_df.shape[1]}"
                        )
                    else:
                        scaled_data = scaler.transform(features_df)
                        input_data = scaled_data.reshape(1, TIMESTEPS, FEATURES)

                        pred_scaled = float(model.predict(input_data).reshape(-1))

                        dummy = np.zeros((1, FEATURES))
                        dummy[0, 3] = pred_scaled  # close price index = 3
                        inv = scaler.inverse_transform(dummy)
                        predicted_price = float(inv[0, 3])

                        current_price = features_df["close"].iloc[-1]
                        delta = predicted_price - current_price

                        prediction_placeholder.metric(
                            "Predicted Price (Next Hour)",
                            f"${predicted_price:,.2f}",
                            f"${delta:,.2f} vs now"
                        )

                        try:
                            log_prediction_to_db(features_df, predicted_price)
                        except:
                            pass

                except Exception as e:
                    prediction_placeholder.error(str(e))


# =====================================================
# PROCESS QUEUE
# =====================================================
new_data_list = []

while not st.session_state.message_queue.empty():
    try:
        msg = st.session_state.message_queue.get_nowait()
        new_data_list.append(msg)
    except queue.Empty:
        break

if len(new_data_list) > 0:
    new_df = pd.DataFrame(new_data_list)
    new_df["time"] = pd.to_datetime(new_df["time"], unit="ms")
    new_df.set_index("time", inplace=True)

    for c in ["open", "high", "low", "close", "volume"]:
        new_df[c] = pd.to_numeric(new_df[c], errors="coerce")

    if st.session_state.live_data.empty:
        st.session_state.live_data = new_df
    else:
        merged = pd.concat([st.session_state.live_data, new_df])
        merged = merged[~merged.index.duplicated(keep="last")]
        st.session_state.live_data = merged.tail(2000)


# =====================================================
# RENDER CHART
# =====================================================
df = st.session_state.live_data.copy()

if not df.empty:
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        increasing_line_color="green",
        decreasing_line_color="red"
    )])

    fig.update_layout(
        title="Live BTC/USDT (1m)",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=500
    )

    chart_placeholder.plotly_chart(fig, use_container_width=True)
    data_grid_placeholder.dataframe(df.tail(20).sort_index(ascending=False))
else:
    chart_placeholder.info("Waiting for live data...")


# =====================================================
# AUTO REFRESH
# =====================================================
time.sleep(1)
st.rerun()
