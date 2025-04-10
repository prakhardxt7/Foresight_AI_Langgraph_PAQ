# File: training/train_forecasting_model.py

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

# Load datasets
nykaa_df = pd.read_csv("data/Nykaa_Dataset.csv")
comp_df = pd.read_csv("data/Competitor_Dataset.csv")

print("✅ Loaded Nykaa dataset:", nykaa_df.shape)
print("✅ Loaded Competitor dataset:", comp_df.shape)

# Merge datasets
df = pd.merge(
    nykaa_df, comp_df,
    on=["Date", "Region", "Product_Name"],
    suffixes=("", "_comp"),
    how="left"
)
print("🔄 After merge:", df.shape)

# Drop duplicates
df.drop_duplicates(inplace=True)

# Convert date and extract Month and Year
df["Date"] = pd.to_datetime(df["Date"])
df["Month"] = df["Date"].dt.month
df["Year"] = df["Date"].dt.year
df.sort_values("Date", inplace=True)

# Encode categorical features
label_encoders = {}
object_cols = df.select_dtypes(include="object").columns.tolist()
for col in object_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Ensure target exists
if "Sales_Units" not in df.columns:
    raise ValueError("❌ 'Sales_Units' column is missing from the dataset.")

# Define features and target
X = df.drop(columns=["Sales_Units", "Date"])
y = df["Sales_Units"]

print("📐 Feature set shape:", X.shape)
print("📐 Target shape:", y.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgb_model.pkl")
joblib.dump(label_encoders, "models/label_encoders.pkl")

print("✅ Model trained and saved to 'models/'")
