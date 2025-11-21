import streamlit as st
import json
from sqlalchemy import text

def log_prediction_to_db(features_df, prediction_value):
    """
    Logs the input features and the model's prediction to our PostgreSQL database.
    """
    try:
        # 1. Prepare data for logging
        features_json = features_df.to_json(orient='split')
        pred_float = float(prediction_value)

        # 2. Initialize connection
        conn = st.connection("postgresql", type="sql")

        # 3. Write to DB
        with conn.session as s:
            sql = text("""
                INSERT INTO predictions_log (features_json, predicted_value)
                VALUES (:features, :pred)
            """)
            s.execute(sql, params={"features": features_json, "pred": pred_float})
            s.commit()
            
        print(f"Successfully logged prediction: {pred_float}")

    except Exception as e:
        print(f"Database Log Error: {e}")