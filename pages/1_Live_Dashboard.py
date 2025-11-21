import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
import requests
import joblib
import time
from streamlit_autorefresh import st_autorefresh

# --- Safe imports ---
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
st_autorefresh(interval=60 * 1000, key="auto")

st.title("Real-Time BTC/USD Prediction Engine")
st.caption("Data Source: Kraken Public API • Live 1-Minute Candles")

# ====================== KRAKEN DATA (CACHED & SAFE) ======================
@st.cache_data(ttl=55, show_spinner=False)
def fetch_kraken_data() -> pd.DataFrame:
    url = "https://api.kraken.com/0/public/OHLC"
    params = {"pair": "XBTUSD", "interval": 1}
    
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if data.get("error"):
            st.error(f"Kraken: {data['error']}")
            return pd.DataFrame()

        pair_key = next((k for k in data["result"] if k != "last"), None)
        if not pair_key:
            return pd.DataFrame()

        candles = data["result"][pair_key]
        df = pd.DataFrame(candles, columns=[
            "time", "open", "high", "low", "close", "vwap", "volume", "count"
        ])
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
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

# ====================== FETCH DATA ======================
df = fetch_kraken_data()

# ====================== LAYOUT ======================
col1, col2 = st.columns([1, 2])

with col1:
    if not df.empty:
        chart_df = df.tail(60)
        fig = go.Figure(go.Candlestick(
            x=chart_df.index,
            open=chart_df["open"], high=chart_df["high"],
            low=chart_df["low"], close=chart_df["close"],
            increasing_line_color="#00ff88", decreasing_line_color="#ff3366"
        ))
        fig.update_layout(
            title="Live BTC/USD (1m)",
            template="plotly_dark",
            height=520,
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig, width='stretch')
        st.info(f"Loaded {len(df)} candles | Last: ${df['close'].iloc[-1]:,.2f}")
    else:
        st.warning("Loading market data...")

with col2:
    st.subheader("AI Forecast")

    if df.empty:
        st.info("Waiting for data...")
    elif len(df) < TIMESTEPS:
        st.warning(f"Need {TIMESTEPS} candles • Have {len(df)}")
    else:
        if st.button("Predict Next Hour Close", type="primary", use_container_width=True):
            if not model or not scaler:
                st.error("Models not loaded.")
            else:
                with st.spinner("Analyzing + predicting..."):
                    # Sentiment
                    sentiment = 0.0
                    try:
                        headlines = fetch_company_news("CRYPTO")
                        if headlines:
                            sentiment, _ = get_sentiment(headlines)
                    except:
                        pass
                    st.metric("Live News Sentiment", f"{sentiment:+.4f}")

                    # === FIX: Match EXACT column names from training ===
                    seq = df.tail(TIMESTEPS).copy()
                    seq = seq.rename(columns={
                        'open': 'Open',
                        'high': 'High',
                        'low': 'Low',
                        'close': 'Close',
                        'volume': 'Volume'
                    })
                    seq = seq[['Open', 'High', 'Low', 'Close', 'Volume']]
                    seq['Sentiment'] = sentiment

                    # Predict
                    X = scaler.transform(seq.values).reshape(1, TIMESTEPS, FEATURES)
                    pred_scaled = model.predict(X, verbose=0)[0][0]

                    # Inverse transform
                    dummy = np.zeros((1, FEATURES))
                    dummy[0, 3] = pred_scaled  # Close is index 3
                    predicted_price = scaler.inverse_transform(dummy)[0, 3]

                    current = df["close"].iloc[-1]
                    change = predicted_price - current
                    pct = change / current * 100

                    st.metric(
                        "Predicted Price (+1h)",
                        f"${predicted_price:,.2f}",
                        f"{change:+.2f} ({pct:+.2f}%)"
                    )

                    # Log
                    try:
                        log_prediction_to_db(seq, predicted_price)
                        st.success("Prediction logged!")
                    except:
                        pass

# ====================== RAW DATA ======================
with st.expander("View Real-Time Data Feed"):
    if not df.empty:
        st.dataframe(df.tail(15).sort_index(ascending=False), width='stretch')