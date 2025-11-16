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

# Keep your original imports for news & sentiment
from src.data_ingestion.news_fetcher import fetch_company_news
from src.ml.sentiment import get_sentiment
from src.ml.preprocessing import TIMESTEPS, FEATURES

import yfinance as yf
import datetime

# ---------------------------
# Price Poller (replaces Binance websocket)
# ---------------------------
class PricePoller:
    """
    Polls Yahoo Finance (yfinance) for the latest 1-minute BTC-USD candle
    and pushes a simple dict into the provided queue.
    """
    def __init__(self, message_queue, symbol="BTC-USD", poll_interval=15):
        self.queue = message_queue
        self.symbol = symbol
        self.poll_interval = poll_interval
        self._stopped = False

    def stop(self):
        self._stopped = True

    def _fetch_latest_minute(self):
        """
        Fetch last 3 minutes at 1m interval to avoid empty frames,
        then return the last available row as a dict.
        """
        try:
            # Use UTC window that covers at least 2-3 minutes
            end = datetime.datetime.utcnow()
            start = end - datetime.timedelta(minutes=5)
            df = yf.download(self.symbol, start=start, end=end, interval="1m", progress=False)
            if df is None or df.empty:
                return None

            last = df.iloc[-1]
            ts = int(pd.to_datetime(df.index[-1]).tz_localize(None).timestamp() * 1000)
            return {
                "time": ts,
                "open": float(last["Open"]),
                "high": float(last["High"]),
                "low": float(last["Low"]),
                "close": float(last["Close"]),
                "volume": float(last["Volume"])
            }
        except Exception as e:
            # Keep polling; transient network issues might happen
            print("PricePoller fetch error:", e)
            return None

    def start_polling(self):
        while not self._stopped:
            item = self._fetch_latest_minute()
            if item is not None:
                try:
                    # Put latest price into queue (non-blocking)
                    self.queue.put_nowait(item)
                except queue.Full:
                    # If queue full, drop the update (we keep only recent)
                    pass
            time.sleep(self.poll_interval)


# ---------------------------
# 1. SESSION STATE & THREAD INITIALIZATION
# ---------------------------

if "message_queue" not in st.session_state:
    st.session_state.message_queue = queue.Queue(maxsize=2000)

if "live_data" not in st.session_state:
    # index 'time' will be set after we receive real messages
    st.session_state.live_data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

if "poller_thread" not in st.session_state:
    poller = PricePoller(st.session_state.message_queue, symbol="BTC-USD", poll_interval=15)
    t = threading.Thread(target=poller.start_polling, daemon=True)
    add_script_run_ctx(t)
    t.start()
    st.session_state.poller_thread = t
    st.session_state.poller = poller
    print("Price poller thread started.")


# ---------------------------
# 2. LOAD MODELS (CACHED)
# ---------------------------
@st.cache_resource
def load_models():
    """Loads LSTM model and scaler. Return (model, scaler) or (None, None)."""
    try:
        model = tf.keras.models.load_model('models/price_model.h5')
    except Exception as e:
        print("Error loading LSTM model:", e)
        model = None

    try:
        scaler = joblib.load('models/price_scaler.pkl')
    except Exception as e:
        print("Error loading scaler:", e)
        scaler = None

    return model, scaler

model, scaler = load_models()


# ---------------------------
# 3. UI LAYOUT
# ---------------------------
st.title("Live BTC/USDT Price & Prediction Dashboard")

price_col, sentiment_col = st.columns([1, 2])

with price_col:
    chart_placeholder = st.empty()
with sentiment_col:
    st.subheader("On-Demand Prediction")
    predict_button = st.button("Predict Next Hour Price")
    prediction_placeholder = st.empty()
    st.subheader("Live Sentiment (Last 24h)")
    sentiment_placeholder = st.empty()

data_grid_placeholder = st.empty()


# ---------------------------
# 4. PREDICTION LOGIC
# ---------------------------
if predict_button:
    if model is None or scaler is None:
        prediction_placeholder.error("Models are not loaded. Cannot predict.")
    else:
        with st.spinner("Running prediction..."):
            try:
                # Need TIMESTEPS rows
                if len(st.session_state.live_data) < TIMESTEPS:
                    prediction_placeholder.warning(f"Not enough data. Need {TIMESTEPS} points to predict.")
                else:
                    # 2. Get live sentiment (safe fallback to neutral 0.0 if fetch fails)
                    try:
                        headlines = fetch_company_news("CRYPTO")
                        live_sentiment_score, _ = get_sentiment(headlines)
                    except Exception as e:
                        print("Sentiment fetch error:", e)
                        live_sentiment_score = 0.0

                    sentiment_placeholder.write(f"Current News Sentiment: {live_sentiment_score:.4f}")

                    # 3. Prepare features (last TIMESTEPS rows)
                    features_df = st.session_state.live_data.tail(TIMESTEPS)[['open', 'high', 'low', 'close', 'volume']].copy()
                    features_df['sentiment'] = live_sentiment_score

                    # Ensure shape and order match scaler expectation (columns: open, high, low, close, volume, sentiment)
                    if features_df.shape[1] != FEATURES:
                        prediction_placeholder.error(f"Feature mismatch: expected {FEATURES} columns but got {features_df.shape[1]}.")
                    else:
                        # 4. Scale and reshape
                        scaled_data = scaler.transform(features_df)  # (TIMESTEPS, FEATURES)
                        input_data = scaled_data.reshape((1, TIMESTEPS, FEATURES))

                        # 5. Predict (model outputs scaled 'close' value)
                        scaled_prediction = model.predict(input_data)  # Expect shape (1,1) or (1, timesteps, 1) depending on model
                        # Normalize to shape (1,) containing the predicted scaled close
                        if scaled_prediction.ndim == 3:
                            # If model returns sequence, take last element
                            scaled_pred_val = scaled_prediction[0, -1, 0]
                        else:
                            scaled_pred_val = float(np.asarray(scaled_prediction).reshape(-1)[0])

                        # 6. Inverse transform
                        # Build a dummy row in scaled space with zeros and put predicted scaled close at index 3
                        dummy_scaled = np.zeros((1, FEATURES))
                        # scaled values expected order: open, high, low, close, volume, sentiment
                        dummy_scaled[0, 3] = scaled_pred_val  # put into 'close' position
                        inversed = scaler.inverse_transform(dummy_scaled)  # returns shape (1, FEATURES)
                        actual_prediction_value = float(inversed[0, 3])

                        # Current price (last close)
                        current_price = float(features_df['close'].iloc[-1])
                        delta = actual_prediction_value - current_price

                        prediction_placeholder.metric(
                            label="Predicted Price (Next Hour)",
                            value=f"${actual_prediction_value:,.2f}",
                            delta=f"${delta:,.2f} vs current"
                        )

                        # Log to DB (catch exceptions to avoid breaking UI)
                        try:
                            log_prediction_to_db(features_df, actual_prediction_value)
                        except Exception as e:
                            print("Logging error:", e)

            except Exception as e:
                prediction_placeholder.error(f"Prediction error: {e}")


# ---------------------------
# 5. DRAIN QUEUE & UPDATE live_data
# ---------------------------
new_data_list = []
while not st.session_state.message_queue.empty():
    try:
        msg = st.session_state.message_queue.get_nowait()
        new_data_list.append(msg)
    except queue.Empty:
        break

if new_data_list:
    new_df = pd.DataFrame(new_data_list)
    # Convert time -> datetime index
    new_df['time'] = pd.to_datetime(new_df['time'], unit='ms')
    new_df.set_index('time', inplace=True)
    # Ensure columns are lowercased and numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
    # Merge with existing live_data
    if st.session_state.live_data.empty:
        st.session_state.live_data = new_df[['open', 'high', 'low', 'close', 'volume']].copy()
    else:
        combined = pd.concat([st.session_state.live_data, new_df[['open', 'high', 'low', 'close', 'volume']]])
        # drop duplicates by index (time)
        combined = combined[~combined.index.duplicated(keep='last')]
        # keep only the last 2000 rows to limit memory
        st.session_state.live_data = combined.tail(2000)


# ---------------------------
# 6. RENDER CHART & DATA
# ---------------------------
df = st.session_state.live_data.copy()
if not df.empty:
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        increasing_line_color='green',
        decreasing_line_color='red',
    )])
    fig.update_layout(
        title='Live BTC/USDT (1-Minute Klines)',
        xaxis_title='Time',
        yaxis_title='Price (USDT)',
        xaxis_rangeslider_visible=False,
        height=520
    )
    chart_placeholder.plotly_chart(fig, use_container_width=True)

    # show last 10 rows
    data_grid_placeholder.dataframe(df.tail(10).sort_index(ascending=False), use_container_width=True)
else:
    chart_placeholder.info("Waiting for live data...")

# ---------------------------
# 7. AUTO-RERUN (simple live tick)
# ---------------------------
# Sleep a short moment so the UI doesn't busy-loop
time.sleep(1)
st.rerun()
