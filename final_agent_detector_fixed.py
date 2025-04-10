import os
import re
import time
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss, accuracy_score

# Load API Key
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini (1.5 Flash)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

# Prompt Template
router_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
You are a query classification agent. Classify the following query into one or more of the categories:
1. Forecasting → If the query asks for predictions (e.g., future demand, trends, projections)
2. MarketWatcher → If the query asks for current beauty product trends or competitor updates
3. FAQs → If the query is a general question about Nykaa, product usage, or returns

Query: "{query}"

Return only the category names: Forecasting, MarketWatcher, FAQs.
"""
)

# LangChain LLM Chain
query_router = LLMChain(llm=llm, prompt=router_prompt)

def classify_query(query):
    return query_router.run(query).strip()

def decompose_query(query):
    delimiters = [" and ", ",", "\n", " also ", " plus "]
    pattern = "|".join(map(re.escape, delimiters))
    return [q.strip() for q in re.split(pattern, query) if q.strip()]

def detect_agents_involved(query):
    sub_queries = decompose_query(query)
    involved_agents = set()
    for q in sub_queries:
        classification = classify_query(q)
        for agent in ["Forecasting", "MarketWatcher", "FAQs"]:
            if agent in classification:
                involved_agents.add(agent)
    return list(involved_agents)

# Load and use the same input file
dataset_path = "multi_intent_query_dataset.csv"
df = pd.read_csv(dataset_path)

# Ensure 'predicted_labels' column exists
if "predicted_labels" not in df.columns:
    df["predicted_labels"] = None

# Find next 5 unprocessed rows
unprocessed_indices = df[df["predicted_labels"].isnull()].index[:3]

if len(unprocessed_indices) == 0:
    print("🎉 All queries have already been classified!")

    # Optional: Evaluate only when all predictions are complete
    y_true = df["labels"].apply(eval).tolist()
    y_pred = df["predicted_labels"].apply(eval).tolist()

    mlb = MultiLabelBinarizer(classes=["FAQs", "Forecasting", "MarketWatcher"])
    y_true_bin = mlb.fit_transform(y_true)
    y_pred_bin = mlb.transform(y_pred)

    print("\n📊 Evaluation Results:")
    print("Micro Precision:", precision_score(y_true_bin, y_pred_bin, average='micro'))
    print("Micro Recall:", recall_score(y_true_bin, y_pred_bin, average='micro'))
    print("Micro F1 Score:", f1_score(y_true_bin, y_pred_bin, average='micro'))
    print("Hamming Loss:", hamming_loss(y_true_bin, y_pred_bin))
    print("Subset Accuracy:", accuracy_score(y_true_bin, y_pred_bin))
else:
    for idx in unprocessed_indices:
        query = df.loc[idx, "query"]
        print(f"🔍 Processing Query #{idx + 1}: {query}")
        prediction = detect_agents_involved(query)
        df.at[idx, "predicted_labels"] = prediction
        print(f"✅ Prediction: {prediction}")
        print("-" * 50)
        time.sleep(2)  # Small delay to avoid flooding API

    # Save back to the same dataset
    df.to_csv(dataset_path, index=False)
    print(f"💾 Saved predictions for {len(unprocessed_indices)} new queries to the same file.")

    remaining = df["predicted_labels"].isnull().sum()
    print(f"📌 Remaining queries to classify: {remaining}")
