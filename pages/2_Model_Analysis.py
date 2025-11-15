import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Import our sentiment module
from src.ml.sentiment import get_sentiment, load_sentiment_model

st.title("Model Performance & Analysis")

# --- 1. Sentiment Model (FinBERT) ---
st.header("Sentiment Analysis (FinBERT)")
st.markdown("""
To meet our success criterion of >80% accuracy, we rejected generic tools like TextBlob, which are "ill-suited" for financial text.[8, 25]
We instead use **FinBERT** (`ProsusAI/finbert`), a state-of-the-art model pre-trained on a massive financial corpus.[26, 27, 24, 28, 29] It achieves ~89% accuracy out-of-the-box.
""")

# Load the model (this will be cached)
with st.spinner("Loading FinBERT model... (This may take a moment on first run)"):
    load_sentiment_model()

st.subheader("Live FinBERT Analysis")
test_headlines = [
    
    "Tesla (TSLA) stock rises on strong Q4 delivery numbers.",
    "The Fed announced an aggressive rate hike, sparking market fears.",
    "Apple (AAPL) profits are flat year-over-year.",
    "Analysts are bullish on the new tech IPO.",
    "The company's earnings report was a disaster."
]

df = pd.DataFrame(test_headlines, columns=["Headline"])

if st.button("Run Sentiment Analysis on Examples"):
    with st.spinner("Analyzing..."):
        avg_score, detailed_sentiments = get_sentiment(df["Headline"].tolist())
        
        # Parse the detailed results
        df['Label'] = [s['label'] for s in detailed_sentiments]
        df['Confidence'] = [f"{s['score']:.2%}" for s in detailed_sentiments]
        
        st.dataframe(df, use_container_width=True)
        st.metric("Average Sentiment of Batch", f"{avg_score:.4f}")

# --- 2. Price Prediction Model (LSTM) ---
st.header("Price Prediction (Stacked LSTM)")
st.markdown("""
Our price model is a **Multivariate Stacked LSTM**. It was trained on 2 years of
hourly BTC/USDT data and uses 6 features:
1.  `Open`
2.  `High`
3.  `Low`
4.  `Close`
5.  `Volume`
6.  `Sentiment Score` (Simulated during training, live during prediction)

It uses a 60-hour (timestep) lookback window to predict the price for the next hour.
""")

st.subheader("Model Training History (Illustrative)")
st.markdown("*(This chart shows illustrative data, as the 'history' object is not persisted after training. In a full MLOps pipeline, this would be logged with MLflow or W&B.)*")

# Create a fake loss plot
epoch = list(range(1, 51))
train_loss = [1.0/x + 0.05 + (0.1 / x) for x in epoch]
val_loss = [1.0/x + 0.1 + (0.05 / x) for x in epoch]

loss_fig = go.Figure()
loss_fig.add_trace(go.Scatter(x=epoch, y=train_loss, mode='lines', name='Training Loss'))
loss_fig.add_trace(go.Scatter(x=epoch, y=val_loss, mode='lines', name='Validation Loss'))
loss_fig.update_layout(
    title="Model Training & Validation Loss Over Epochs",
    xaxis_title="Epoch",
    yaxis_title="Mean Squared Error (Scaled)"
)
st.plotly_chart(loss_fig, use_container_width=True)