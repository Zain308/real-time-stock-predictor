import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

@st.cache_resource
def load_sentiment_model():
    """
    Loads and caches the FinBERT sentiment analysis pipeline.
    This uses @st.cache_resource to ensure the model is loaded only once.
    """
    print("Loading FinBERT model...")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    nlp_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    print("FinBERT model loaded.")
    return nlp_pipeline

def get_sentiment(headlines: list[str]):
    """
    Analyzes a list of headlines and returns their sentiment scores.
    
    Returns:
        A tuple: (average_sentiment_score, list_of_detailed_sentiments)
    """
    if not headlines:
        return 0.0, # Return 0.0 average and empty list if no headlines

    nlp = load_sentiment_model()
    sentiments = nlp(headlines)
    
    score_map = {'positive': 1.0, 'negative': -1.0, 'neutral': 0.0}
    
    # Calculate a single score for each headline, weighted by confidence
    numeric_scores = [score_map[s['label']] * s['score'] for s in sentiments]
    
    # Calculate the average sentiment
    if not numeric_scores:
        return 0.0,
        
    average_sentiment = sum(numeric_scores) / len(numeric_scores)
    
    return average_sentiment, sentiments

if __name__ == "__main__":
    # Test the sentiment module
    # This will download the model the first time you run it.
    print("Testing FinBERT model...")
    nlp = load_sentiment_model()
    
    test_headlines = [
        "Tesla (TSLA) stock rises on strong Q4 delivery numbers.",
        "The Fed announced an aggressive rate hike, sparking market fears.",
        "Apple (AAPL) profits are flat year-over-year.",
        "Analysts are bullish on the new tech IPO.",
        "The company's earnings report was a disaster."
    ]
    
    avg_score, detailed_sentiments = get_sentiment(test_headlines)
    
    print(f"\nTest Headlines:\n{test_headlines}")
    print(f"\nDetailed Sentiments:\n{detailed_sentiments}")
    print(f"\nAverage Sentiment Score: {avg_score:.4f}")