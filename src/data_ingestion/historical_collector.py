import os
import pandas as pd
from binance.us.client import Client
from dotenv import load_dotenv


def fetch_historical_data():
    """
    Fetches historical kline data from Binance.US and saves it to a CSV file.
    """
    load_dotenv()
    api_key = os.environ.get("BINANCE_API_KEY")
    api_secret = os.environ.get("BINANCE_API_SECRET")

    if not api_key or not api_secret:
        print("Error: BINANCE_API_KEY or BINANCE_API_SECRET not found in .env file or Streamlit secrets.")
        return

    # Initialize Binance.US Client
    client = Client(api_key, api_secret)

    print("⏳ Fetching historical data from Binance.US...")

    # Binance US supports BTCUSDT, interval: 1 hour
    klines_generator = client.get_historical_klines_generator(
        "BTCUSDT",
        Client.KLINE_INTERVAL_1HOUR,
        "2 years ago UTC"
    )

    klines = list(klines_generator)

    # Binance US returns 12 fields:
    columns = [
        "Open Time", "Open", "High", "Low", "Close", "Volume",
        "Close Time", "Quote Asset Volume",
        "Number of Trades", "Taker Buy Base Volume",
        "Taker Buy Quote Volume", "Ignore"
    ]

    # Create DataFrame
    df = pd.DataFrame(klines, columns=columns)

    # Convert timestamps
    df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms")
    df["Close Time"] = pd.to_datetime(df["Close Time"], unit="ms")

    # Convert numeric fields
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Index: Open Time
    df.set_index("Open Time", inplace=True)

    # Final DataFrame: 5 price features
    df_final = df[["Open", "High", "Low", "Close", "Volume"]]

    # Output path
    output_path = "data/historical_prices.csv"
    df_final.to_csv(output_path)

    print(f"✅ Successfully saved {len(df_final)} rows to {output_path}")


if __name__ == "__main__":
    if not os.path.exists('data'):
        os.makedirs('data')

    fetch_historical_data()
