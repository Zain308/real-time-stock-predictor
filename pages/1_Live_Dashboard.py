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

# Import our custom modules
from src.data_ingestion.live_streamer import BinanceStreamer
from src.data_ingestion.news_fetcher import fetch_company_news
from src.ml.sentiment import get_sentiment
from src.ml.preprocessing import TIMESTEPS, FEATURES

# --- 1. SESSION STATE & THREAD INITIALIZATION ---
# This is the core of the real-time architecture
# We must store the queue and thread in session_state to ensure they persist
# across Streamlit's script reruns. 

if "message_queue" not in st.session_state:
    st.session_state.message_queue = queue.Queue(maxsize=1000)

if "live_data" not in st.session_state:
    st.session_state.live_data = pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])

if "streamer" not in st.session_state:
    # Initialize the streamer and start it in a background thread
    streamer = BinanceStreamer(st.session_state.message_queue)
    st.session_state.streamer = streamer
    
    t = threading.Thread(target=streamer.start_stream, daemon=True)
    add_script_run_ctx(t) # Attach Streamlit's context to the thread [5, 23, 6]
    t.start()
    st.session_state.websocket_thread = t
    print("WebSocket thread started.")

# --- 2. LOAD MODELS (CACHED) ---
# Caching these models is critical for performance

@st.cache_resource
def load_models():
    """Loads the trained LSTM model and the scaler."""
    try:
        model = tf.keras.models.load_model('models/price_model.h5')
        scaler = joblib.load('models/price_scaler.pkl')
        print("Models loaded successfully.")
        return model, scaler
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None

model, scaler = load_models()

# --- 3. UI LAYOUT ---
st.title("Live BTC/USDT Price & Prediction Dashboard")

# Create placeholders for dynamic content
price_col, sentiment_col = st.columns([1, 2])

with price_col:
    chart_placeholder = st.empty() # Placeholder for the live chart [11, 12]
with sentiment_col:
    st.subheader("On-Demand Prediction")
    predict_button = st.button("Predict Next Hour Price")
    prediction_placeholder = st.empty() # Placeholder for the prediction result
    st.subheader("Live Sentiment (Last 24h)")
    sentiment_placeholder = st.empty() # Placeholder for live sentiment

data_grid_placeholder = st.empty() # Placeholder for the raw data table

# --- 4. PREDICTION LOGIC ---

if predict_button:
    if model is None or scaler is None:
        prediction_placeholder.error("Models are not loaded. Cannot predict.")
    else:
        with st.spinner("Running prediction..."):
            try:
                # 1. Get last 60 data points (timesteps)
                if len(st.session_state.live_data) < TIMESTEPS:
                    prediction_placeholder.warning(f"Not enough data. Need {TIMESTEPS} minutes of data to predict.")
                else:
                    # 2. Get live sentiment
                    headlines = fetch_company_news("CRYPTO")
                    live_sentiment_score, _ = get_sentiment(headlines)
                    sentiment_placeholder.metric("Current News Sentiment", f"{live_sentiment_score:.4f}")

                    # 3. Prepare features
                    # Get last 60 rows of OHLCV data
                    features_df = st.session_state.live_data.tail(TIMESTEPS)[['open', 'high', 'low', 'close', 'volume']]

                    # Add the *current* sentiment score as the 6th feature
                    features_df['sentiment'] = live_sentiment_score

                    # 4. Scale and Reshape
                    scaled_data = scaler.transform(features_df)
                    input_data = scaled_data.reshape((1, TIMESTEPS, FEATURES)) # Reshape to (1, 60, 6)

                    # 5. Predict
                    scaled_prediction = model.predict(input_data)

                    # 6. Inverse Transform (CRITICAL CORRECTION)
                    # We must "reverse" the scaling to get a real dollar amount.
                    # We create a dummy array of 6 features (shape (1,6))
                    # and place our prediction in the 'close' column (index 3).
                    dummy_array = np.zeros((1, FEATURES))
                    dummy_array = scaled_prediction # Put prediction in the 'close' slot

                    # Now, inverse_transform the whole dummy array
                    inversed_result = scaler.inverse_transform(dummy_array)

                    # The real dollar value is at index 3
                    actual_prediction_value = inversed_result

                    current_price = features_df['close'].iloc[-1]
                    delta = actual_prediction_value - current_price

                    prediction_placeholder.metric(
                        label="Predicted Price (Next Hour)",
                        value=f"${actual_prediction_value:,.2f}",
                        delta=f"${delta:,.2f} vs current"
                    )

                    # --- 7. LOG TO DATABASE ---
                    # This is our new step
                    log_prediction_to_db(features_df, actual_prediction_value)
                    # -------------------------

            except Exception as e:
                prediction_placeholder.error(f"Prediction error: {e}")
                
                # --- 5. LIVE DATA UPDATE & RERUN LOOP ---

# Drain the queue and update the live data DataFrame
new_data_list = []
while not st.session_state.message_queue.empty():
    msg = st.session_state.message_queue.get()
    new_data_list.append(msg)

if new_data_list:
    new_df = pd.DataFrame(new_data_list)
    new_df['time'] = pd.to_datetime(new_df['time'], unit='ms')
    new_df.set_index('time', inplace=True)
    
    st.session_state.live_data = pd.concat([st.session_state.live_data, new_df])
    # Keep only the last 1000 records to prevent memory overflow
    st.session_state.live_data = st.session_state.live_data.tail(1000)

# 6. Redraw UI Placeholders
df = st.session_state.live_data
if not df.empty:
    # Draw Candlestick Chart
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close']
    )])
    fig.update_layout(
        title='Live BTC/USDT (1-Minute Klines)',
        xaxis_title='Time',
        yaxis_title='Price (USDT)',
        xaxis_rangeslider_visible=False,
        height=500
    )
    chart_placeholder.plotly_chart(fig, use_container_width=True)
    
    # Update data grid
    data_grid_placeholder.dataframe(df.tail(10).sort_index(ascending=False), use_container_width=True)

# 7. Auto-rerun script to create live "tick"
time.sleep(1)
st.rerun()