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

# Logger import handling
try:
    from src.db_logger import log_prediction_to_db
except ImportError:
    from src.logging import log_prediction_to_db

from src.data_ingestion.news_fetcher import fetch_company_news
from src.ml.sentiment import get_sentiment
from src.ml.preprocessing import TIMESTEPS, FEATURES


# -----------------------------------------------------------
# 1. CONFIGURATION
# -----------------------------------------------------------
st.set_page_config(layout="wide", page_title="Live Crypto Dashboard")

# Refresh every 60 seconds
st_autorefresh(interval=60 * 1000, key="data_refresher")


# -----------------------------------------------------------
# 2. LIVE DATA FETCHER (SAFE + CLEAN)
# -----------------------------------------------------------
@st.cache_data(ttl=60)
def load_live_data(symbol="BINANCE:BTCUSDT"):

    try:
        api_key = st.secrets.get("FINNHUB_API_KEY") or os.getenv("FINNHUB_API_KEY")
        if not api_key:
            st.error("Missing FINNHUB_API_KEY. Add it in secrets.toml or environment.")
            return pd.DataFrame()

        client = finnhub.Client(api_key=api_key)

        to_ts = int(time.time())
        from_ts = to_ts - (60 * 120)  # last 120 minutes

        res = client.crypto_candles(symbol, '1', from_ts, to_ts)

        if res.get("s") != "ok" or "t" not in res or not res["t"]:
            return pd.DataFrame()

        df = pd.DataFrame({
            "time": pd.to_datetime(res["t"], unit="s"),
            "open": res["o"],
            "high": res["h"],
            "low": res["l"],
            "close": res["c"],
            "volume": res["v"],
        })

        df.set_index("time", inplace=True)
        df = df[~df.index.duplicated(keep="last")].sort_index()

        return df

    except Exception as e:
        st.error(f"Live data error: {e}")
        return pd.DataFrame()


# -----------------------------------------------------------
# 3. LOAD MODELS SAFELY
# -----------------------------------------------------------
@st.cache_resource
def load_models():
    try:
        model = tf.keras.models.load_model("models/price_model.h5")
        scaler = joblib.load("models/price_scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None, None


model, scaler = load_models()


# -----------------------------------------------------------
# 4. DASHBOARD
# -----------------------------------------------------------
st.title("Live BTC/USDT Prediction Engine")
st.caption("Data updates every 60 seconds from Finnhub")

df = load_live_data()

col1, col2 = st.columns([1, 3])


# -----------------------------------------------------------
# CANDLESTICK CHART
# -----------------------------------------------------------
with col1:
    if df.empty:
        st.info("Waiting for live data...")
    else:
        fig = go.Figure(
            data=[go.Candlestick(
                x=df.index,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350"
            )]
        )
        fig.update_layout(
            title="Real-Time Market Data (1m candles)",
            height=500,
            xaxis_rangeslider_visible=False,
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------
# PREDICTION PANEL
# -----------------------------------------------------------
with col2:
    st.subheader("AI Prediction")

    if df.empty:
        st.warning("Loading data...")
    elif len(df) < TIMESTEPS:
        st.warning(f"Not enough candles. Need {TIMESTEPS}, have {len(df)}")
    else:
        predict_btn = st.button("Predict Next Hour")

        if predict_btn:
            if model is None or scaler is None:
                st.error("Model files missing.")
            else:
                with st.spinner("Running prediction..."):
                    try:
                        # Fetch sentiment
                        try:
                            headlines = fetch_company_news("CRYPTO")
                            sentiment_score, _ = get_sentiment(headlines)
                        except:
                            sentiment_score = 0.0

                        st.metric("News Sentiment", f"{sentiment_score:.4f}")

                        # Prepare input window
                        input_df = df.tail(TIMESTEPS).copy()
                        input_df = input_df[["open", "high", "low", "close", "volume"]]
                        input_df["sentiment"] = sentiment_score

                        # Scale
                        scaled = scaler.transform(input_df)
                        model_input = scaled.reshape((1, TIMESTEPS, FEATURES))

                        # Predict
                        prediction_scaled = model.predict(model_input)
                        pred_scaled = float(prediction_scaled[0][0])

                        # Proper inverse scaling
                        dummy = np.zeros((1, FEATURES))
                        dummy[0][3] = pred_scaled   # assuming index 3 = close

                        inverse = scaler.inverse_transform(dummy)
                        predicted_price = float(inverse[0][3])

                        current_price = float(df["close"].iloc[-1])
                        delta = predicted_price - current_price

                        st.metric(
                            label="Predicted Close (Next Hour)",
                            value=f"${predicted_price:,.2f}",
                            delta=f"{delta:+.2f}"
                        )

                        # Log to DB safely
                        try:
                            log_prediction_to_db(input_df, predicted_price)
                        except:
                            pass

                    except Exception as e:
                        st.error(f"Prediction error: {e}")


# -----------------------------------------------------------
# RAW DATA TABLE
# -----------------------------------------------------------
with st.expander("Show Raw Data"):
    st.dataframe(df.sort_index(ascending=False), use_container_width=True)
