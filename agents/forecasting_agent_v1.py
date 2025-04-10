import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
from difflib import get_close_matches
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import os
import re
from dotenv import load_dotenv

# === Load environment ===
load_dotenv()

# === Google Gemini Setup ===
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

# === Clarify Prompt ===
clarify_prompt = PromptTemplate.from_template("""
You are a smart demand forecasting assistant.
You must understand vague user queries and restructure them clearly.

- Convert vague time expressions like 'next week', 'July', 'September', etc. to YYYY-MM-DD format.
  * For example:
    - "Next week" → The start date of the next week (e.g., 2025-04-06).
    - "July" → The first day of July (e.g., 2025-07-01).
    - "next month" → The first day of the next month (e.g., 2025-05-01).
  * For other ambiguous date references (like 'fall season'), normalize them to the most relevant date range.
- Normalize regions: 'South India' to 'South', 'West India' to 'West', 'Tamil Nadu' to 'South', etc.

Restructure the following user query:
User Query: {query}

Return in this format:
Product: <product>
Region: <region>
Time: <yyyy-mm-dd>
""")
clarifier_chain = clarify_prompt | llm

# === Enhanced Forecast Summary Prompt ===
summary_prompt = PromptTemplate.from_template("""
You are a senior forecasting analyst at Nykaa. Based on the forecast data, provide actionable business insights tailored for regional strategy and operations teams.

🧾 Forecast Input:
- Product: {product}
- Region: {region}
- Forecasted Units: {units}

Please return a structured 3-part strategic recommendation:

1. 📌 **Inventory Planning Suggestion**
- How should Nykaa handle stock ordering for this product?
- Should the team adopt conservative ordering, or build buffer stock?
- Consider using JIT (Just-in-Time) or safety stock logic based on demand.
- Discuss logistics impact, shelf-life issues, and restock time if relevant.

2. 📊 **Pricing or Promotion Ideas**
- Suggest discount strategies or price optimizations.
- Recommend bundling options or loyalty program activations.
- Propose digital promotions (e.g., quizzes, gamified sampling, trials).
- Include geo-targeted influencer or content creator collaborations.

3. 📅 **Demand Pattern Interpretation**
- What might explain this level of forecast? (Is it seasonal, competitor-related, trend-driven?)
- Analyze regional customer behavior or cosmetic preferences.
- Consider whether this is a data limitation or a market signal.
- Recommend what team should investigate next.

Return your answer in this format:
📌 Inventory Planning Suggestion: [...]
📊 Pricing or Promotion Ideas: [...]
📅 Demand Pattern Interpretation: [...]
""")
summary_chain = summary_prompt | llm


class ForecastingAgent:
    def __init__(
        self,
        data_path="training/Nykaa_Enriched_Dataset.csv",
        model_path="xgboost_forecasting_model.pkl",
        product_encoder_path="product_encoder.pkl",
        region_encoder_path="region_encoder.pkl"
    ):
        self.df = pd.read_csv(data_path)
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.model = joblib.load(model_path)
        self.product_encoder = joblib.load(product_encoder_path)
        self.region_encoder = joblib.load(region_encoder_path)
        self.all_products = self.df["Product_Name"].unique().tolist()

    def _clarify_query(self, query: str):
        response = clarifier_chain.invoke({"query": query}).content
        product_match = re.search(r"Product:\s*(.*)", response)
        region_match = re.search(r"Region:\s*(.*)", response)
        date_match = re.search(r"Time:\s*(.*)", response)

        product = product_match.group(1).strip() if product_match else None
        raw_region = region_match.group(1).strip() if region_match else None
        date = date_match.group(1).strip() if date_match else None

        # Clean region: Extract only from known valid values
        valid_regions = ["North", "South", "East", "West", "Central"]
        region = next((r for r in valid_regions if r.lower() in raw_region.lower()), None) if raw_region else None

        return product, region, date

    def _parse_date(self, date_str: str):
        try:
            if "next week" in date_str.lower():
                return datetime.today() + relativedelta(weeks=1, weekday=0)
            elif "next month" in date_str.lower():
                return datetime.today() + relativedelta(months=1, day=1)
            elif re.match(r"^[A-Za-z]+$", date_str.strip()):
                today = datetime.today()
                month_num = datetime.strptime(date_str.strip(), "%B").month
                return datetime(today.year, month_num, 1)
            return parse(date_str, fuzzy=True)
        except:
            return None

    def _fuzzy_product_match(self, input_name: str):
        matches = get_close_matches(input_name.lower(), [p.lower() for p in self.all_products], n=1, cutoff=0.4)
        if matches:
            return next((p for p in self.all_products if p.lower() == matches[0]), None)
        return None

    def forecast(self, user_query: str) -> str:
        product, region, date_str = self._clarify_query(user_query)

        if not product or not region:
            return "❌ Could not understand product or region from your query."

        product = self._fuzzy_product_match(product)
        if not product:
            return f"❌ Product '{product}' not found."

        try:
            region_encoded = self.region_encoder.transform([region])[0]
        except:
            return f"❌ Region '{region}' not recognized. Valid: North, South, East, West, Central"

        hist = self.df[(self.df["Product_Name"] == product) & (self.df["Region"] == region)]
        if hist.empty:
            return f"❌ No history for '{product}' in region '{region}'."

        hist = hist.sort_values("Date")
        if hist.shape[0] < 3:
            return f"❌ Not enough data for forecasting '{product}' in '{region}'."

        last_row = hist.iloc[-1].copy()

        if date_str:
            parsed_date = self._parse_date(date_str)
            if not parsed_date:
                return "❌ Could not parse forecast date. Use YYYY-MM-DD."
            last_row["Month"] = parsed_date.month
            last_row["Day"] = parsed_date.day
            last_row["Weekday"] = parsed_date.weekday()
            last_row["Year"] = parsed_date.year
        else:
            parsed_date = last_row["Date"]

        past = hist[hist["Date"] < parsed_date].tail(3)
        if past.shape[0] < 1:
            return "❌ Not enough past records to calculate lag features."

        try:
            prev_sales = past.iloc[-1]["Sales_Units"]
            prev_spend = past.iloc[-1]["Marketing_Spend"]
            rolling_avg = past["Sales_Units"].mean()

            product_encoded = self.product_encoder.transform([product])[0]
            region_encoded = self.region_encoder.transform([region])[0]
            marketing_spend = prev_spend
            month, year, day = parsed_date.month, parsed_date.year, parsed_date.day
            weekday = parsed_date.weekday()
            is_weekend = 1 if weekday >= 5 else 0

            X_pred = np.array([
                product_encoded, region_encoded, marketing_spend,
                month, year, day, weekday, is_weekend,
                prev_sales, prev_spend, rolling_avg
            ]).reshape(1, -1)

            forecast_units = self.model.predict(X_pred)[0]

            # Summary with Gemini
            summary = summary_chain.invoke({
                "product": product,
                "region": region,
                "units": int(round(forecast_units))
            }).content.strip()

            return (
                f"🔮 **Forecast Result**\n"
                f"📦 Product: {product}\n"
                f"🗺️ Region: {region}\n"
                f"📅 Date: {parsed_date.strftime('%Y-%m-%d')}\n"
                f"📈 Expected Sales: {int(round(forecast_units))} units\n\n"
                f"{summary}"
            )
        except Exception as e:
            return f"❌ Forecasting failed: {str(e)}"


# === Example test ===
if __name__ == "__main__":
    agent = ForecastingAgent()
    test_query = "Forecast Nykaa Cosmetics Get Cheeky Blush Stick in North next week"
    print(agent.forecast(test_query))
