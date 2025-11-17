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

# Imports for our ML models and news
from src.data_ingestion.news_fetcher import fetch_company_news
from src.ml.sentiment import get_sentiment
from src.ml.preprocessing import TIMESTEPS, FEATURES

# Imports for the new YFinance Poller
import yfinance as yf
import datetime

# ---------------------------
# Price Poller (Replaces Binance WebSocket)
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
        print("PricePoller initialized.")

    def stop(self):
        self._stopped = True

    def _fetch_latest_minute(self):
        """
        Fetch last 5 minutes at 1m interval to avoid empty frames,
        then return the last available row as a dict.
        """
        try:
            # Use a 5-minute window to ensure we get data
            end = datetime.datetime.utcnow()
            start = end - datetime.timedelta(minutes=5)
            df = yf.download(self.symbol, start=start, end=end, interval="1m", progress=False)
            
            if df is None or df.empty:
                return None

            last_row = df.iloc[-1]
            # Convert index (timestamp) to UTC milliseconds
            ts = int(pd.to_datetime(df.index[-1]).tz_localize(None).timestamp() * 1000)
            
            return {
                "time": ts,
                "open": float(last_row["Open"]),
                "high": float(last_row["High"]),
                "low": float(last_row["Low"]),
                "close": float(last_row["Close"]),
                "volume": float(last_row["Volume"])
            }
        except Exception as e:
            print(f"PricePoller fetch error: {e}")
            return None

    def start_polling(self):
        """
        Main background loop for the poller thread.
        """
        print("Polling thread started.")
        while not self._stopped:
            item = self._fetch_latest_minute()
            
            if item is not None:
                try:
                    # Put latest price into queue (non-blocking)
                    self.queue.put_nowait(item)
                except queue.Full:
                    pass # If queue is full, we drop this update
            
            # Wait for the poll interval
            time.sleep(self.poll_interval)
        print("Polling thread stopped.")

# ---------------------------
# 1. SESSION STATE & THREAD INITIALIZATION
# ---------------------------

if "message_queue" not in st.session_state:
    st.session_state.message_queue = queue.Queue(maxsize=2000)

if "live_data" not in st.session_state:
    st.session_state.live_data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

if "poller_thread" not in st.session_state:
    # Initialize and start the new YFinance poller
    poller = PricePoller(st.session_state.message_queue, symbol="BTC-USD", poll_interval=15)
    t = threading.Thread(target=poller.start_polling, daemon=True)
    add_script_run_ctx(t) [4]
    t.start()
    st.session_state.poller_thread = t
    st.session_state.poller = poller
    print("Price poller thread registered in session state.")

# ---------------------------
# 2. LOAD MODELS (CACHED)
# ---------------------------
@st.cache_resource
def load_models():
    """Loads LSTM model and scaler. Returns (model, scaler) or (None, None)."""
    model, scaler = None, None
    try:
        model = tf.keras.models.load_model('models/price_model.h5')
    except Exception as e:
        print(f"Error loading LSTM model: {e}")
    try:
        scaler = joblib.load('models/price_scaler.pkl')
    except Exception as e:
        print(f"Error loading scaler: {e}")
    
    return model, scaler

model, scaler = load_models()

# ---------------------------
# 3. UI LAYOUT
# ---------------------------
st.title("Live BTC/USDT Price & Prediction Dashboard")

col1, col2 = st.columns([1, 2])

with col1:
    chart_placeholder = st.empty()

with col2:
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
                # Check if we have enough data (TIMESTEPS = 60)
                if len(st.session_state.live_data) < TIMESTEPS:
                    prediction_placeholder.warning(f"Not enough data. Need {TIMESTEPS} data points to predict.")
                else:
                    # Get live sentiment
                    live_sentiment_score = 0.0
                    try:
                        headlines = fetch_company_news("CRYPTO")
                        live_sentiment_score, _ = get_sentiment(headlines)
                    except Exception as e:
                        print(f"Sentiment fetch error: {e}")
                    
                    sentiment_placeholder.write(f"Current News Sentiment: {live_sentiment_score:.4f}")

                    # Prepare features (last 60 rows)
                    features_df = st.session_state.live_data.tail(TIMESTEPS)[['open', 'high', 'low', 'close', 'volume']].copy()
                    features_df['sentiment'] = live_sentiment_score

                    if features_df.shape![2]= FEATURES:
                        prediction_placeholder.error(f"Feature mismatch: Expected {FEATURES}, got {features_df.shape[2]}.")
                    else:
                        # Scale and reshape
                        scaled_data = scaler.transform(features_df)
                        input_data = scaled_data.reshape((1, TIMESTEPS, FEATURES))

                        # Predict
                        scaled_prediction = model.predict(input_data)
                        scaled_pred_val = float(np.asarray(scaled_prediction).reshape(-1))

                        # Inverse transform to get real dollar value
                        dummy_scaled = np.zeros((1, FEATURES))
                        target_index = 3 # 'close' is at index 3
                        dummy_scaled[0, target_index] = scaled_pred_val
                        
                        inversed = scaler.inverse_transform(dummy_scaled)
                        actual_prediction_value = float(inversed[0, target_index])

                        current_price = float(features_df['close'].iloc[-1])
                        delta = actual_prediction_value - current_price

                        prediction_placeholder.metric(
                            label="Predicted Price (Next Hour)",
                            value=f"${actual_prediction_value:,.2f}",
                            delta=f"${delta:,.2f} vs current"
                        )

                        # Log to DB
                        try:
                            log_prediction_to_db(features_df, actual_prediction_value)
                        except Exception as e:
                            print(f"Logging error: {e}")

            except Exception as e:
                prediction_placeholder.error(f"Prediction error: {e}")

# ---------------------------
# 5. DRAIN QUEUE & UPDATE live_data
# ---------------------------
new_data_list =
while not st.session_state.message_queue.empty():
    try:
        msg = st.session_state.message_queue.get_nowait()
        new_data_list.append(msg)
    except queue.Empty:
        break

if new_data_list:
    new_df = pd.DataFrame(new_data_list)
    new_df['time'] = pd.to_datetime(new_df['time'], unit='ms')
    new_df.set_index('time', inplace=True)
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
    
    if st.session_state.live_data.empty:
        st.session_state.live_data = new_df[['open', 'high', 'low', 'close', 'volume']].copy()
    else:
        combined = pd.concat([st.session_state.live_data, new_df[['open', 'high', 'low', 'close', 'volume']]])
        # Drop duplicates by index (time), keeping the last-received update
        combined = combined[~combined.index.duplicated(keep='last')]
        st.session_state.live_data = combined.tail(2000) # Limit memory

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
        title='Live BTC/USDT (1-Minute Klines from YFinance)',
        xaxis_title='Time',
        yaxis_title='Price (USDT)',
        xaxis_rangeslider_visible=False,
        height=500
    )
    chart_placeholder.plotly_chart(fig, use_container_width=True) [5, 6]
    data_grid_placeholder.dataframe(df.tail(10).sort_index(ascending=False), use_container_width=True)
else:
    chart_placeholder.info("Waiting for live data...")

# ---------------------------
# 7. AUTO-RERUN
# ---------------------------
time.sleep(1)
st.rerun() [6]