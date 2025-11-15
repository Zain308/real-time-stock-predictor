import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from src.ml.preprocessing import preprocess_data, TIMESTEPS, FEATURES

def build_model(input_shape):
    """
    Builds a stacked LSTM model with Dropout.
    """
    model = Sequential()
    
    # Layer 1
    model.add(LSTM(
        units=50,
        return_sequences=True, # Required for stacked LSTMs
        input_shape=input_shape
    ))
    model.add(Dropout(0.2)) # Prevents overfitting
    
    # Layer 2
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Output Layer
    model.add(Dense(units=1)) # Predicts a single value (the next 'Close' price)
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    return model

def train_model():
    """
    Loads preprocessed data and trains the LSTM model.
    Saves the final model artifact.
    """
    # Ensure 'models' directory exists
    if not os.path.exists('models'):
        os.makedirs('models')
        
    print("Starting preprocessing...")
    train_gen, val_gen = preprocess_data()
    
    if train_gen is None:
        print("Preprocessing failed. Halting training.")
        return

    print("Building model...")
    # Input shape is (timesteps, features)
    model = build_model(input_shape=(TIMESTEPS, FEATURES))

    # Define callbacks (e.g., early stopping)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    print("Starting model training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=50, # Set to a lower number (e.g., 5-10) for a quick test
        callbacks=[early_stopping]
    )

    print("Training complete.")
    
    # Save the final model
    model.save('models/price_model.h5')
    print("Model saved to models/price_model.h5")
    
    return history

if __name__ == "__main__":
    # This allows us to run the entire training pipeline from the command line
    train_model()