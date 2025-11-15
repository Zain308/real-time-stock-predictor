import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# --- Constants ---
TIMESTEPS = 60
BATCH_SIZE = 32
FEATURES = 6   # OHLCV + Sentiment


def load_and_prepare_data():
    """
    Loads historical price data, adds a simulated sentiment feature,
    and returns a 6-feature DataFrame for multivariate LSTM training.
    """
    try:
        df = pd.read_csv(
            'data/historical_prices.csv',
            index_col='Open Time',
            parse_dates=True
        )
    except FileNotFoundError:
        print("❌ Error: data/historical_prices.csv not found.")
        print("Run: src/data_ingestion/historical_collector.py first.")
        return None

    # Ensure sorted by time
    df.sort_index(inplace=True)

    # Select the 5 OHLCV price features
    df_clean = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

    # --- Fix Sentiment Feature Bug ---
    # Before: You replaced df_clean completely → breaking your pipeline
    np.random.seed(42)
    sentiment = np.random.uniform(-1, 1, size=len(df_clean))

    # Add sentiment correctly as a NEW COLUMN
    df_clean['Sentiment'] = sentiment

    print(f"Loaded data with {len(df_clean)} rows.")
    print(f"Dataset columns: {df_clean.columns.tolist()}")
    return df_clean


def preprocess_data():
    """
    Loads, splits, scales, and windows the time series data.
    Saves MinMaxScaler model for production use.
    """
    df = load_and_prepare_data()
    if df is None:
        return None, None

    # --- 1. Train / Val / Test Split ---
    n = len(df)
    train_df = df[:int(n * 0.7)]
    val_df = df[int(n * 0.7):int(n * 0.9)]
    test_df = df[int(n * 0.9):]

    # --- 2. Scale (fit only on training to avoid leakage) ---
    scaler = MinMaxScaler()
    scaler.fit(train_df)

    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/price_scaler.pkl')
    print("Scaler saved to models/price_scaler.pkl")

    # Apply scaler to all splits
    train_scaled = scaler.transform(train_df)
    val_scaled = scaler.transform(val_df)
    test_scaled = scaler.transform(test_df)

    # Target is the CLOSE price (index 3)
    target_index = 3

    # --- 3. Create Keras Time-Series Generators ---
    train_generator = TimeseriesGenerator(
        train_scaled,
        train_scaled[:, target_index],
        length=TIMESTEPS,
        batch_size=BATCH_SIZE
    )

    val_generator = TimeseriesGenerator(
        val_scaled,
        val_scaled[:, target_index],
        length=TIMESTEPS,
        batch_size=BATCH_SIZE
    )

    print(f"Train generator: {len(train_generator)} batches")
    print(f"Val generator: {len(val_generator)} batches")

    return train_generator, val_generator


if __name__ == "__main__":
    preprocess_data()
