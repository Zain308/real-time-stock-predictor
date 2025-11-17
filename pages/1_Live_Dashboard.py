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

# Imports for ML models and news
from src.data_ingestion.news_fetcher import fetch_company_news
from src.ml.sentiment import get_sentiment
from src.ml.preprocessing import TIMESTEPS, FEATURES

# Imports for Finnhub Poller
import finnhub
import datetime

# -----------------------------------------------------------
# FINNHUB PRICE POLLER
# -----------------------------------------------------------
class FinnhubPoller:
    def __init__(self, message_queue, symbol="BINANCE:BTCUSDT", poll_interval=15):
        self.queue = message_queue
        self.symbol = symbol
        self.poll_interval = poll_interval
        self._stopped = False
        
        try:
            api_key = st.secrets["FINNHUB_API_KEY"]
            self.finnhub_client = finnhub.Client(api_key=api_key)
            print("FinnhubPoller initialized.")
        except Exception as e:
            print(f"Error initializing Finnhub client: {e}")
            self.finnhub_client = None

    def stop(self):
        self._stopped = True

    def _fetch_latest_minute(self):
        if self.finnhub_client is None:
            return None

        try:
            to_ts = int(time.time())
            from_ts = to_ts - 300  # last 5 minutes

            res = self.finnhub_client.crypto_candles(self.symbol, '1', from_ts, to_ts)

            if res.get("s") != "ok" or not res.get("t"):
                print("FinnhubPoller: No candle data received.")
                return None

            last = -1
            item = {
                "time": int(res["t"][last] * 1000),
                "open": float(res["o"][last]),
                "high": float(res["h"][last]),
                "low": float(res["l"][last]),
                "close": float(res["c"][last]),
                "volume": float(res["v"][last])
            }
            return item

        except Exception as e:
            print(f"FinnhubPoller fetch error: {e}")
            return None

    def start_polling(self):
        print("FinnhubPoller thread started.")
        while not self._stopped:
            item = self._fetch_latest_minute()

            if item:
                try:
                    self.queue.put_nowait(item)
                except queue.Full:
                    pass

            time.sleep(self.poll_interval)

        print("Polling thread stopped.")


# -----------------------------------------------------------
# SESSION STATE INIT
# -----------------------------------------------------------

if "message_queue" not in st.session_state:
    st.session_state.message_queue = queue.Queue(maxsize=2000)

if "live_data" not in st.session_state:
    st.session_state.live_data = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

if "poller_thread" not in st.session_state:
    poller = FinnhubPoller(st.session_state.message_queue, "BINANCE:BTCUSDT", 15)
    t = threading.Thread(target=poller.start_polling, daemon=True)
    add_script_run_ctx(t)
    t.start()
    st.session_state.poller_thread = t
    st.session_state.poller = poller
    print("Price poller thread registered.")


# -----------------------------------------------------------
# LOAD MODELS
# -----------------------------------------------------------
@st.cache_resource
def load_models():
    model, scaler = None, None
    try:
        model = tf.keras.models.load_model("models/price_model.h5")
    except Exception as e:
        print("Model load error:", e)
    try:
        scaler = joblib.load("models/price_scaler.pkl")
    except Exception as e:
        print("Scaler load error:", e)

    return model, scaler

model, scaler = load_models()


# -----------------------------------------------------------
# UI LAYOUT
# -----------------------------------------------------------
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


# -----------------------------------------------------------
# PREDICTION LOGIC
# -----------------------------------------------------------
if predict_button:
    if model is None or scaler is None:
        prediction_placeholder.error("Models not loaded.")
    else:
        with st.spinner("Running prediction..."):
            try:
                if len(st.session_state.live_data) < TIMESTEPS:
                    prediction_placeholder.warning(
                        f"Not enough data. Need {TIMESTEPS} rows."
                    )
                else:
                    try:
                        headlines = fetch_company_news("CRYPTO")
                        live_sentiment_score, _ = get_sentiment(headlines)
                    except:
                        live_sentiment_score = 0.0

                    sentiment_placeholder.write(
                        f"Current News Sentiment: {live_sentiment_score:.4f}"
                    )

                    features_df = st.session_state.live_data.tail(TIMESTEPS).copy()
                    features_df["sentiment"] = live_sentiment_score

                    if features_df.shape[1] != FEATURES:
                        prediction_placeholder.error(
                            f"Feature mismatch: expected {FEATURES}, got {features_df.shape[1]}"
                        )
                    else:
                        scaled_data = scaler.transform(features_df)
                        input_data = scaled_data.reshape((1, TIMESTEPS, FEATURES))

                        scaled_prediction = model.predict(input_data)
                        scaled_pred_val = float(scaled_prediction.reshape(-1))

                        dummy = np.zeros((1, FEATURES))
                        dummy[0, 3] = scaled_pred_val

                        inv = scaler.inverse_transform(dummy)
                        final_price = float(inv[0, 3])

                        current_price = float(features_df["close"].iloc[-1])
                        delta = final_price - current_price

                        prediction_placeholder.metric(
                            label="Predicted Price (Next Hour)",
                            value=f"${final_price:,.2f}",
                            delta=f"${delta:,.2f} vs current"
                        )

                        try:
                            log_prediction_to_db(features_df, final_price)
                        except Exception as e:
                            print("Logging error:", e)

            except Exception as e:
                prediction_placeholder.error(f"Prediction error: {e}")


# -----------------------------------------------------------
# DRAIN QUEUE & UPDATE live_data
# -----------------------------------------------------------
new_data_list = []

while not st.session_state.message_queue.empty():
    try:
        msg = st.session_state.message_queue.get_nowait()
        new_data_list.append(msg)
    except queue.Empty:
        break

if new_data_list:
    new_df = pd.DataFrame(new_data_list)
    new_df["time"] = pd.to_datetime(new_df["time"], unit="ms")
    new_df.set_index("time", inplace=True)

    for col in ["open", "high", "low", "close", "volume"]:
        new_df[col] = pd.to_numeric(new_df[col], errors="coerce")

    if st.session_state.live_data.empty:
        st.session_state.live_data = new_df.copy()
    else:
        combined = pd.concat([st.session_state.live_data, new_df])
        combined = combined[~combined.index.duplicated(keep="last")]
        st.session_state.live_data = combined.tail(2000)


# -----------------------------------------------------------
# RENDER CHART
# -----------------------------------------------------------
df = st.session_state.live_data.copy()

if not df.empty:
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df.index,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                increasing_line_color="green",
                decreasing_line_color="red",
            )
        ]
    )

    fig.update_layout(
        title="Live BTC/USDT (1-Min Candles from Finnhub)",
        xaxis_title="Time",
        yaxis_title="Price (USDT)",
        xaxis_rangeslider_visible=False,
        height=500,
    )

    chart_placeholder.plotly_chart(fig, use_container_width=True)
    data_grid_placeholder.dataframe(
        df.tail(10).sort_index(ascending=False),
        use_container_width=True
    )
else:
    chart_placeholder.info("Waiting for live data...")


# -----------------------------------------------------------
# AUTO-RERUN
# -----------------------------------------------------------
time.sleep(1)
st.rerun()
