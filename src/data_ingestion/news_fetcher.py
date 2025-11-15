import os
import finnhub
from dotenv import load_dotenv
from datetime import datetime, timedelta

def fetch_company_news(symbol="CRYPTO"):
    """
    Fetches company news for a given symbol from Finnhub.
    """
    load_dotenv()
    api_key = os.environ.get("FINNHUB_API_KEY")

    if not api_key:
        print("Error: FINNHUB_API_KEY not found in.env file.")
        return # Return an empty list to prevent errors

    try:
        finnhub_client = finnhub.Client(api_key=api_key) [1, 2]

        # Get today's and yesterday's date for the query
        today = datetime.now().strftime('%Y-%m-%d')
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

        # We fetch news for 'Crypto' as a general category
        news_list = finnhub_client.company_news(symbol, _from=yesterday, to=today)

        # We only care about the headlines for sentiment analysis
        headlines = [item['headline'] for item in news_list if item['headline']]
        return headlines

    except Exception as e:
        print(f"Error fetching news from Finnhub: {e}")
        return # Return an empty list on failure

if __name__ == "__main__":
    # Test the function
    print("Testing Finnhub news fetcher...")
    headlines = fetch_company_news()
    if headlines:
        print(f"Successfully fetched {len(headlines)} headlines:")
        for i, h in enumerate(headlines[:5]):
            print(f"{i+1}. {h}")
    else:
        print("No headlines fetched. Check API key and Finnhub permissions.")