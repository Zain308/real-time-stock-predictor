import os
import pandas as pd
from binance.client import Client
from dotenv import load_dotenv

def fetch_historical_data():
    """
    Fetches historical kline data from Binance and saves it to a CSV file.
    """
    # Load API keys from .env file [1, 2]
    load_dotenv()
    api_key = os.environ.get("BINANCE_API_KEY")
    api_secret = os.environ.get("BINANCE_API_SECRET")

    # Check if keys are loaded
    if not api_key or not api_secret:
        print("Error: BINANCE_API_KEY or BINANCE_API_SECRET not found in .env file.")
        print("Please ensure your .env file is set up correctly.")
        return

    client = Client(api_key, api_secret)  # [3] ← removed invalid [3] syntax

    # Fetch 2 years of hourly data for BTCUSDT
    print("Fetching historical data... This may take a moment.")
    klines_generator = client.get_historical_klines_generator(
        "BTCUSDT",
        Client.KLINE_INTERVAL_1HOUR,
        "2 years ago UTC"
    )  # [4] ← removed invalid [4] syntax

    klines = list(klines_generator)

    # Define all 12 columns based on Binance API documentation [5, 6, 7]
    columns = [
        "Open Time", "Open", "High", "Low", "Close", "Volume",
        "Close Time", "Quote Asset Volume", "Number of Trades",
        "Taker Buy Base Volume", "Taker Buy Quote Volume", "Ignore"
    ]

    df = pd.DataFrame(klines, columns=columns)

    # --- Data Cleaning and Processing ---
    
    # 1. Select only the columns we need for training
    df_clean = df.copy()  # fixed df] syntax error
    
    # 2. Convert timestamps to datetime objects [6, 7]
    df_clean["Open Time"] = pd.to_datetime(df_clean["Open Time"], unit='ms')  # fixed incorrect usage
    
    # 3. Convert price and volume columns to numeric types
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df_clean[col] = pd.to_numeric(df_clean[col])
        
    # 4. Set the datetime as the index
    df_clean.set_index('Open Time', inplace=True)
    
    # Save to the data/ directory (which is in .gitignore)
    output_path = 'data/historical_prices.csv'
    df_clean.to_csv(output_path)
    
    print(f"Successfully fetched and saved {len(df_clean)} records to {output_path}")

if __name__ == "__main__":
    # This allows us to run this file directly to get our data
    # Ensure the 'data' directory exists
    if not os.path.exists('data'):
        os.makedirs('data')
    fetch_historical_data()
