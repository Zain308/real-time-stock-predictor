import streamlit as st
import time
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import plotly.graph_objects as go
import finnhub
import os
from streamlit_autorefresh import st_autorefresh

# Import our other modules
from src.logging import log_prediction_to_db
from src.data_ingestion.news_fetcher import fetch_company_news
from src.ml.sentiment import get_sentiment
from src.ml.preprocessing import TIMESTEPS, FEATURES

# ---------------------------
# 1. AUTO-REFRESHER
# ---------------------------
st_autorefresh(interval=60 * 1000, key="data_refresher")

# ---------------------------
# 2. FINNHUB DATA LOADER
# ---------------------------
@st.cache_data(ttl=60)
def load_live_data(symbol="BINANCE:BTCUSDT"):
    """
    Fetches the last 2 hours of 1-minute candle data from Finnhub.
    """
    print("Fetching new data from Finnhub REST API...")

    try:
        # Load API key
        api_key = st.secrets.get("FINNHUB_API_KEY")
        if not api_key:
            api_key = os.environ.get("FINNHUB_API_KEY")

        if not api_key:
            st.error("Finnhub API key missing in secrets or environment variables.")
            return pd.DataFrame()

        finnhub_client = finnhub.Client(api_key=api_key)

        # Fetch last 120 minutes (2 hours)
        to_ts = int(time.time())
        from_ts = to_ts - (60 * 120)

        res = finnhub_client.crypto_candles(symbol, '1', from_ts, to_ts)

        if res.get('s') != 'ok' or 't' not in res or not res['t']:
            st.error("Error fetching data from Finnhub.")
            return pd.DataFrame()

        df = pd.DataFrame({
            'time': [pd.to_datetime(t, unit='s') for t in res['t']],
            'open': res['o'],
            'high': res['h'],
            'low': res['l'],
            'close': res['c'],
            'volume': res['v']
        })

        df.set_index('time', inplace=True)
        df = df[~df.index.duplicated(keep='last')].sort_index()

        return df

    except Exception as e:
        st.error(f"Error in load_live_data: {e}")
        return pd.DataFrame()

# ---------------------------
# 3. LOAD MODELS (CACHED)
# ---------------------------
@st.cache_resource
def load_models():
    model, scaler = None, None

    try:
        model = tf.keras.models.load_model('models/price_model.h5')
    except Exception as e:
        print(f"Error loading LSTM model: {e}")

    try:
        scaler = joblib.load('models/price_scaler.pkl')
    except Exception as e:
        print(f"Error loading scaler: {e}")

    if model and scaler:
        print("Models loaded successfully.")

    return model, scaler

model, scaler = load_models()

# ---------------------------
# 4. UI LAYOUT
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
# 5. LOAD LIVE DATA
# ---------------------------
df = load_live_data()

# ---------------------------
# 6. PREDICTION LOGIC
# ---------------------------
if predict_button:

    if model is None or scaler is None:
        prediction_placeholder.error("Models are not loaded. Cannot predict.")

    elif len(df) < TIMESTEPS:
        prediction_placeholder.warning(f"Not enough data. Need {TIMESTEPS} rows, but only {len(df)} loaded.")

    else:
        with st.spinner("Running prediction..."):
            try:
                # Fetch sentiment
                live_sentiment_score = 0.0
                try:
                    headlines = fetch_company_news("CRYPTO")
                    live_sentiment_score, _ = get_sentiment(headlines)
                except Exception as e:
                    print(f"Sentiment fetch error: {e}")

                sentiment_placeholder.write(f"Current News Sentiment: {live_sentiment_score:.4f}")

                # Build feature frame
                features_df = df.tail(TIMESTEPS)[['open', 'high', 'low', 'close', 'volume']].copy()
                features_df['sentiment'] = live_sentiment_score

                # ---- FIXED FEATURE DIM CHECK ----
                if features_df.shape[1] != FEATURES:
                    prediction_placeholder.error(
                        f"Feature mismatch: Expected {FEATURES}, got {features_df.shape[1]}"
                    )

                else:
                    # Scale and reshape
                    scaled_data = scaler.transform(features_df)
                    input_data = scaled_data.reshape((1, TIMESTEPS, FEATURES))

                    # Predict
                    scaled_prediction = model.predict(input_data)
                    scaled_pred_val = float(np.asarray(scaled_prediction).reshape(-1))

                    # Inverse transform
                    dummy_scaled = np.zeros((1, FEATURES))
                    target_index = 3  # 'close'
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

                    # Log prediction
                    try:
                        log_prediction_to_db(features_df, actual_prediction_value)
                    except Exception as e:
                        print(f"Logging error: {e}")

            except Exception as e:
                prediction_placeholder.error(f"Prediction error: {e}")

# ---------------------------
# 7. RENDER CHART & DATA
# ---------------------------
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
        title="Live BTC/USDT (Auto-refresh every 60 seconds)",
        xaxis_title="Time",
        yaxis_title="Price (USDT)",
        xaxis_rangeslider_visible=False,
        height=500
    )

    chart_placeholder.plotly_chart(fig, use_container_width=True)
    data_grid_placeholder.dataframe(
        df.tail(10).sort_index(ascending=False),
        use_container_width=True
    )

else:
    chart_placeholder.error("Could not load live data from Finnhub. Please check your API key.")
        