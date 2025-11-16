import streamlit as st
import queue
import threading
import time
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import plotly.graph_objects as go
import websocket
import json
from streamlit.runtime.scriptrunner import add_script_run_ctx

# Custom modules
from src.logging import log_prediction_to_db
from src.data_ingestion.news_fetcher import fetch_company_news
from src.ml.sentiment import get_sentiment
from src.ml.preprocessing import TIMESTEPS, FEATURES


# ---------------------------------------------------------
# 1. YAHOO FINANCE REALTIME STREAMER (REPLACES BINANCE)
# ---------------------------------------------------------

class YahooStreamer:
    def __init__(self, message_queue):
        self.message_queue = message_queue
        self.ws = None
        self.url = "wss://streamer.finance.yahoo.com/"

    def on_message(self, ws, message):
        try:
            msg = json.loads(message)

            if "data" not in msg:
                return

            data = msg["data"][0]

            self.message_queue.put({
                "time": int(time.time() * 1000),
                "open": data.get("open", 0),
                "high": data.get("high", 0),
                "low": data.get("low", 0),
                "close": data.get("price", 0),
                "volume": data.get("volume", 0)
            })

        except Exception as e:
            print("Message parse error:", e)

    def on_open(self, ws):
        print("Yahoo Finance WebSocket Connected.")
        # Subscribe to BTC-USD
        subscribe_msg = json.dumps({
            "subscribe": ["BTC-USD"]
        })
        ws.send(subscribe_msg)

    def on_error(self, ws, error):
        print("WebSocket Error:", error)

    def on_close(self, ws, close_status_code, close_msg):
        print("Yahoo WebSocket closed.")

    def start_stream(self):
        self.ws = websocket.WebSocketApp(
            self.url,
            on_message=self.on_message,
            on_open=self.on_open,
            on_error=self.on_error,
            on_close=self.on_close
        )
        self.ws.run_forever()


# ---------------------------------------------------------
# 2. STREAMLIT SESSION STATE & THREAD CREATION
# ---------------------------------------------------------

if "message_queue" not in st.session_state:
    st.session_state.message_queue = queue.Queue(maxsize=2000)

if "live_data" not in st.session_state:
    st.session_state.live_data = pd.DataFrame(
        columns=["time", "open", "high", "low", "close", "volume"]
    )

if "stream_started" not in st.session_state:
    streamer = YahooStreamer(st.session_state.message_queue)
    t = threading.Thread(target=streamer.start_stream, daemon=True)
    add_script_run_ctx(t)
    t.start()

    st.session_state.stream_started = True
    print("Yahoo Finance Streaming Thread Started")


# ---------------------------------------------------------
# 3. LOAD MODELS
# ---------------------------------------------------------

@st.cache_resource
def load_models():
    try:
        model = tf.keras.models.load_model("models/price_model.h5")
        scaler = joblib.load("models/price_scaler.pkl")
        print("Models loaded successfully.")
        return model, scaler
    except Exception as e:
        print("Model loading error:", e)
        return None, None

model, scaler = load_models()


# ---------------------------------------------------------
# 4. UI DESIGN
# ---------------------------------------------------------

st.title("Live BTC Price & Prediction Dashboard (Yahoo Finance API)")

price_col, sentiment_col = st.columns([1, 2])

with price_col:
    chart_placeholder = st.empty()

with sentiment_col:
    st.subheader("Price Prediction")
    predict_button = st.button("Predict Next Hour Price")
    prediction_placeholder = st.empty()
    st.subheader("Live Sentiment (Last 24h)")
    sentiment_placeholder = st.empty()

data_grid_placeholder = st.empty()


# ---------------------------------------------------------
# 5. PREDICTION LOGIC
# ---------------------------------------------------------

if predict_button:
    if model is None or scaler is None:
        prediction_placeholder.error("Model not loaded.")
    else:
        with st.spinner("Predicting..."):
            try:
                df = st.session_state.live_data

                if len(df) < TIMESTEPS:
                    prediction_placeholder.warning(
                        f"Not enough data. Need {TIMESTEPS} points."
                    )
                else:
                    headlines = fetch_company_news("CRYPTO")
                    sentiment_score, _ = get_sentiment(headlines)

                    sentiment_placeholder.metric(
                        "Live Sentiment Score",
                        f"{sentiment_score:.4f}"
                    )

                    features_df = df.tail(TIMESTEPS)[
                        ["open", "high", "low", "close", "volume"]
                    ]
                    features_df["sentiment"] = sentiment_score

                    scaled = scaler.transform(features_df)
                    inp = scaled.reshape(1, TIMESTEPS, FEATURES)

                    pred_scaled = model.predict(inp)

                    dummy = np.zeros((1, FEATURES))
                    dummy[0] = pred_scaled

                    pred_actual = scaler.inverse_transform(dummy)[0][3]

                    current_price = features_df["close"].iloc[-1]
                    delta = pred_actual - current_price

                    prediction_placeholder.metric(
                        "Predicted BTC Price (Next Hour)",
                        f"${pred_actual:,.2f}",
                        f"{delta:,.2f}"
                    )

                    log_prediction_to_db(features_df, pred_actual)

            except Exception as e:
                prediction_placeholder.error(f"Prediction failed: {e}")


# ---------------------------------------------------------
# 6. PROCESS NEW STREAM DATA
# ---------------------------------------------------------

new_data = []
while not st.session_state.message_queue.empty():
    new_data.append(st.session_state.message_queue.get())

if new_data:
    df_new = pd.DataFrame(new_data)
    df_new["time"] = pd.to_datetime(df_new["time"], unit='ms')
    df_new.set_index("time", inplace=True)

    st.session_state.live_data = pd.concat(
        [st.session_state.live_data, df_new]
    ).tail(1000)


# ---------------------------------------------------------
# 7. CHART + DATA TABLE
# ---------------------------------------------------------

df = st.session_state.live_data

if not df.empty:
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"]
    )])

    fig.update_layout(
        title="Real-Time BTC/USDT (Yahoo Finance API)",
        xaxis_rangeslider_visible=False,
        height=500
    )

    chart_placeholder.plotly_chart(fig, use_container_width=True)

    data_grid_placeholder.dataframe(df.tail(20))


# Auto-refresh
time.sleep(1)
st.rerun()
