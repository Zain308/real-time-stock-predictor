import streamlit as st
import json
from sqlalchemy import text

def log_prediction_to_db(features_df, prediction_value):
    """
    Logs the input features and the model's prediction to our PostgreSQL database.
    """
    try:
        # 1. Prepare data for logging
        # Convert the features DataFrame to a JSON string
        features_json = features_df.to_json(orient='split')

        # Ensure prediction_value is a standard Python float
        pred_float = float(prediction_value)

        # 2. Initialize connection. This automatically reads our secrets.toml
        conn = st.connection("postgresql", type="sql") [5, 6]

        # 3. Use conn.session for write operations (INSERT, UPDATE)
        # This is the correct, senior-level way to write data.
        # conn.query() is for read-only (SELECT) [7, 8]
        with conn.session as s:
            sql = text("""
                INSERT INTO predictions_log (features_json, predicted_value)
                VALUES (:features, :pred)
            """)

            # We use parameterized queries (:features, :pred) to prevent SQL injection
            s.execute(sql, params={
                "features": features_json,
                "pred": pred_float
            })
            s.commit()

        print(f"Successfully logged prediction: {pred_float}")

    except Exception as e:
        # Log to the Streamlit console if the database write fails
        print(f"Database Log Error: {e}")
        st.error(f"Database Log Error: {e}")