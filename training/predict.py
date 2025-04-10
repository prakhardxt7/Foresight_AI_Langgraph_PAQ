import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os

# === Config ===
DATASET_PATH = "training/Nykaa_Enriched_Dataset.csv"
MODEL_PATH = "xgboost_forecasting_model.pkl"
PRODUCT_ENCODER_PATH = "product_encoder.pkl"
REGION_ENCODER_PATH = "region_encoder.pkl"

# === Load assets ===
if not all(os.path.exists(p) for p in [DATASET_PATH, MODEL_PATH, PRODUCT_ENCODER_PATH, REGION_ENCODER_PATH]):
    raise FileNotFoundError("❌ Model or data files not found. Run training first.")

df = pd.read_csv(DATASET_PATH)
df['Date'] = pd.to_datetime(df['Date'])

model = joblib.load(MODEL_PATH)
product_encoder = joblib.load(PRODUCT_ENCODER_PATH)
region_encoder = joblib.load(REGION_ENCODER_PATH)

# === Prediction Function ===
def predict_sales(product_name: str, region: str, date_str: str):
    try:
        date = pd.to_datetime(date_str)
        year, month, day = date.year, date.month, date.day
        weekday = date.weekday()
        is_weekend = 1 if weekday >= 5 else 0

        # Get past records for this product-region
        past_data = df[
            (df['Product_Name'] == product_name) &
            (df['Region'] == region) &
            (df['Date'] < date)
        ].sort_values(by='Date')

        # === Validation Checks ===
        if past_data.empty:
            return f"❌ No data available for product '{product_name}' in region '{region}'. Check name or region."

        if past_data.shape[0] < 3:
            last_date = df[
                (df['Product_Name'] == product_name) &
                (df['Region'] == region)
            ]['Date'].max().strftime("%Y-%m-%d")
            return f"❌ Not enough historical data for prediction. Try a date after {last_date}."

        # Last known values
        prev_sales = past_data.iloc[-1]['Sales_Units']
        prev_marketing_spend = past_data.iloc[-1]['Marketing_Spend']
        rolling_avg_sales = past_data['Sales_Units'].tail(3).mean()
        marketing_spend = prev_marketing_spend  # assume same spend for now

        # Encode inputs
        product_encoded = product_encoder.transform([product_name])[0]
        region_encoded = region_encoder.transform([region])[0]

        # Feature vector
        features = np.array([
            product_encoded, region_encoded, marketing_spend,
            month, year, day, weekday, is_weekend,
            prev_sales, prev_marketing_spend, rolling_avg_sales
        ]).reshape(1, -1)

        # Prediction
        predicted_units = model.predict(features)[0]
        return round(predicted_units, 2)

    except Exception as e:
        return f"❌ Prediction failed: {str(e)}"

# === Example Test Case ===
if __name__ == "__main__":
    forecast = predict_sales(
        product_name="Nykaa Cosmetics Wanderlust Sun Protection SPF 50 Body Lotion (50 gm)",
        region="North",
        date_str="2030-12-15"
    )
    print(f"🔮 Forecasted Sales Units: {forecast}")

