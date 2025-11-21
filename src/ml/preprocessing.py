import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# --- Constants ---
TIMESTEPS = 60      # 60 hours of lookback
BATCH_SIZE = 32
FEATURES = 6        # 5 price features (OHLCV) + 1 sentiment feature

def load_and_prepare_data():
    """
    Loads historical price data and simulates sentiment to create
    a 6-feature DataFrame for multivariate training.
    """
    try:
        df = pd.read_csv('data/historical_prices.csv', index_col='Open Time', parse_dates=True)
    except FileNotFoundError:
        print("Error: data/historical_prices.csv not found.")
        print("Please run src/data_ingestion/historical_collector.py first.")
        return None

    # Ensure data is sorted by time
    df.sort_index(inplace=True)
    
    # 1. Start with our 5 price features
    df_clean = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    # --- Sentiment Feature Simulation ---
    # 2. ADD the 6th feature. 
    np.random.seed(42) # for reproducibility
    df_clean = np.random.uniform(-1, 1, size=len(df_clean))
    # ------------------------------------
    
    print(f"Loaded data with {len(df_clean)} rows. Features: {df_clean.columns.tolist()}")
    return df_clean

def preprocess_data():
    """
    Loads, splits, scales, and windows the time series data.
    Saves the scaler for later use in production.
    """
    df = load_and_prepare_data()
    if df is None:
        return None, None

    # --- 1. Split Data (70% train, 20% val, 10% test) ---
    n = len(df)
    train_df = df[0:int(n*0.7)]
    val_df = df[int(n*0.7):int(n*0.9)]
    test_df = df[int(n*0.9):]

    # --- 2. Scale Data ---
    scaler = MinMaxScaler()
    scaler.fit(train_df)
    
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(scaler, 'models/price_scaler.pkl')
    print("Price scaler saved to models/price_scaler.pkl")

    train_scaled = scaler.transform(train_df)
    val_scaled = scaler.transform(val_df)
    test_scaled = scaler.transform(test_df)

    # --- 3. Create Time Series Generators ---
    # The target is the 'Close' price (index 3 of our 6 features)
    target_index = 3 
    
    train_generator = TimeseriesGenerator(
        train_scaled,
        train_scaled[:, target_index], # Target is the 'Close' column
        length=TIMESTEPS,
        batch_size=BATCH_SIZE
    )
    
    val_generator = TimeseriesGenerator(
        val_scaled,
        val_scaled[:, target_index],
        length=TIMESTEPS,
        batch_size=BATCH_SIZE
    )
    
    print(f"Train generator created with {len(train_generator)} batches.")
    print(f"Validation generator created with {len(val_generator)} batches.")

    return train_generator, val_generator

if __name__ == "__main__":
    # This allows us to run this file directly to test preprocessing
    preprocess_data()