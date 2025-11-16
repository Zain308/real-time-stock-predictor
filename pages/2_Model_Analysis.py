import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Import FinBERT modules
from src.ml.sentiment import get_sentiment, load_sentiment_model


# --------------------------------------
# PAGE TITLE
# --------------------------------------
st.title("📊 Model Performance & Analysis")


# =========================================================
# 1. SENTIMENT MODEL (FinBERT)
# =========================================================
st.header("🧠 Sentiment Analysis – FinBERT (Financial NLP Model)")
st.markdown("""
We do **not** use generic NLP tools such as TextBlob because they are *not reliable* in financial contexts.  
We instead use **FinBERT (`ProsusAI/finbert`)**, a transformer model trained on financial texts, achieving ~89% accuracy.
""")

# Load sentiment model
with st.spinner("Loading FinBERT model..."):
    try:
        sentiment_model = load_sentiment_model()
    except Exception as e:
        st.error(f"Error loading sentiment model: {e}")
        sentiment_model = None


# --------------------------------------
# TEST HEADLINES
# --------------------------------------
st.subheader("Test FinBERT on Example Headlines")

test_headlines = [
    "Tesla (TSLA) stock rises on strong Q4 delivery numbers.",
    "The Fed announced an aggressive rate hike, sparking market fears.",
    "Apple (AAPL) profits are flat year-over-year.",
    "Analysts are bullish on the new tech IPO.",
    "The company's earnings report was a disaster."
]

df = pd.DataFrame(test_headlines, columns=["Headline"])


if st.button("Run Sentiment Analysis"):
    if sentiment_model is None:
        st.error("Sentiment model not loaded.")
    else:
        with st.spinner("Analyzing sentiments..."):
            try:
                avg_score, detailed_sentiments = get_sentiment(
                    df["Headline"].tolist()
                )

                # Add outputs into the DataFrame
                df["Label"] = [d.get("label", "N/A") for d in detailed_sentiments]
                df["Confidence"] = [
                    f"{d.get('score', 0):.2%}" for d in detailed_sentiments
                ]

                st.dataframe(df, use_container_width=True)

                st.metric(
                    label="Average Sentiment Score",
                    value=f"{avg_score:.4f}"
                )

            except Exception as e:
                st.error(f"Error during sentiment analysis: {e}")


# =========================================================
# 2. PRICE MODEL (LSTM)
# =========================================================
st.header("📈 Price Prediction Model – Multivariate Stacked LSTM")

st.markdown("""
This is a **Stacked LSTM model** trained using 6 features:

1. `Open`
2. `High`
3. `Low`
4. `Close`
5. `Volume`
6. `Sentiment Score` (Simulated during training, live added in real-time)

It uses a **60-hour sliding window** to predict the **next-hour BTC price**.
""")


# --------------------------------------
# TRAINING HISTORY (illustrative)
# --------------------------------------
st.subheader("Training Loss Curve (Illustrative Example)")

st.markdown("""
⚠️ *This is an illustrative plot.*  
In a real MLOps pipeline, training metrics would be logged via **MLflow**, **Weights & Biases**, or **Neptune**.
""")

epochs = list(range(1, 51))
train_loss = [1/x + 0.05 for x in epochs]
val_loss = [1/x + 0.08 for x in epochs]

loss_fig = go.Figure()
loss_fig.add_trace(go.Scatter(
    x=epochs, y=train_loss, mode="lines", name="Training Loss"
))
loss_fig.add_trace(go.Scatter(
    x=epochs, y=val_loss, mode="lines", name="Validation Loss"
))

loss_fig.update_layout(
    title="Training vs Validation Loss Across Epochs",
    xaxis_title="Epoch",
    yaxis_title="MSE (Scaled)",
    template="plotly_white",
    height=450
)

st.plotly_chart(loss_fig, use_container_width=True)
