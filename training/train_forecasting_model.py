import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# === Config ===
DATASET_PATH = "training/Nykaa_Enriched_Dataset.csv"
MODEL_PATH = "models/xgboost_forecasting_model.pkl"

# === Load dataset ===
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"❌ Dataset not found at {DATASET_PATH}.")

df = pd.read_csv(DATASET_PATH)
df['Date'] = pd.to_datetime(df['Date'])

# === Feature Engineering ===
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Weekday'] = df['Date'].dt.weekday
df['Is_Weekend'] = df['Weekday'].apply(lambda x: 1 if x >= 5 else 0)

# Label Encoding
product_encoder = LabelEncoder()
df['Product_Name_Encoded'] = product_encoder.fit_transform(df['Product_Name'])

region_encoder = LabelEncoder()
df['Region_Encoded'] = region_encoder.fit_transform(df['Region'])

# Sort for lag feature creation
df = df.sort_values(by=['Product_Name', 'Region', 'Date'])

# Lag features
df['Prev_Sales'] = df.groupby(['Product_Name', 'Region'])['Sales_Units'].shift(1)
df['Prev_Marketing_Spend'] = df.groupby(['Product_Name', 'Region'])['Marketing_Spend'].shift(1)

# === Rolling Average Fix with clean groupby (no warning) ===
df['Rolling_Avg_Sales'] = (
    df.groupby(['Product_Name', 'Region'])['Sales_Units']
    .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
)

# Drop rows where lag features are NaN
df = df.dropna(subset=['Prev_Sales', 'Prev_Marketing_Spend'])

# === Prepare data ===
features = [
    'Product_Name_Encoded', 'Region_Encoded', 'Marketing_Spend',
    'Month', 'Year', 'Day', 'Weekday', 'Is_Weekend',
    'Prev_Sales', 'Prev_Marketing_Spend', 'Rolling_Avg_Sales'
]
target = 'Sales_Units'

X = df[features]
y = df[target]

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# === Train Model ===
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    eval_metric='rmse',
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6
)
model.fit(X_train, y_train)
print("✅ Model training complete.")

# === Evaluate ===
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n===============================")
print(f"📊 RMSE on Test Set: {rmse:.2f}")
print("===============================\n")

# === Save model & encoders ===
joblib.dump(model, MODEL_PATH)
joblib.dump(product_encoder, 'models/product_encoder.pkl')
joblib.dump(region_encoder, 'models/region_encoder.pkl')
print("✅ Model and encoders saved.\n")
