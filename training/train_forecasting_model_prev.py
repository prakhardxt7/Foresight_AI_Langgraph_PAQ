import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

# Load enriched dataset
df = pd.read_csv("data/Nykaa_Enriched_Dataset.csv")
df["Date"] = pd.to_datetime(df["Date"])
df["Month"] = df["Date"].dt.month
df["Week"] = df["Date"].dt.isocalendar().week.astype(int)

# Encode categorical columns
label_encoders = {}
for col in ['Region', 'Category', 'Sub_Category']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and target
feature_cols = [
    'Product_ID', 'Region', 'Category', 'Sub_Category',
    'Price_At_Time', 'Marketing_Spend', 'Social_Media_Influence_Score',
    'Festival_Seasonality_Adjustment', 'Customer_Rating', 'Rolling_Avg_Sales',
    'Comp_Avg_Price', 'Comp_Avg_Sales_Units', 'Comp_Avg_Marketing_Spend',
    'Comp_Avg_Social_Influence', 'Month', 'Week'
]
X = df[feature_cols]
y = df["Sales_Units"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBRegressor(n_estimators=30, max_depth=3, learning_rate=0.1)
model.fit(X_train, y_train)

# Save model and encoders
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgb_demand_model_with_competitor.pkl")
joblib.dump(label_encoders, "models/label_encoders.pkl")
print("✅ Model and encoders saved to 'models/'")
