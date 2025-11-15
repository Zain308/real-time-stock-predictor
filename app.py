import streamlit as st

# Set wide layout and page title
st.set_page_config(layout="wide", page_title="Real-Time Crypto & Sentiment Predictor")

st.title("Real-Time Stock Price & Sentiment Predictor")
st.markdown("""
Welcome to the Real-Time Stock Price & Sentiment Predictor.

This application demonstrates a production-level MLOps pipeline, integrating:
- **Real-time data ingestion** from the Binance WebSocket.
- **Live sentiment analysis** using FinBERT on news from the Finnhub API.
- **Multivariate time series forecasting** with a trained LSTM model.

**Select a page from the sidebar to begin:**
- **Live Dashboard:** View the real-time price chart and make on-demand predictions.
- **Model Analysis:** See the performance of our predictive models.
""")