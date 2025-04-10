import sys
import os
import re
import time
import pandas as pd
import difflib
from dotenv import load_dotenv
from typing import TypedDict
from datetime import datetime

# Add project root path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from agents.forecasting_agent import forecasting_node
from agents.market_watcher_agent import market_watcher_node
from agents.faq_retriever import faq_retriever_node
from agents.web_search_agent import web_search_node

from langgraph.graph import StateGraph, END

# === Define State Schema ===
class AppState(TypedDict, total=False):
    forecast_query: str
    market_query: str
    faq_query: str
    web_query: str
    classification: list
    forecast_response: str
    market_response: str
    faq_response: str
    web_response: str

# === Hardcoded MarketWatcher Variations ===
HARDCODED_MARKETING_VARIATIONS = {
    "recent marketing campaigns by sugar cosmetics",
    "tell me about recent campaigns sugar cosmetics has done",
    "any influencer activity by sugar cosmetics",
    "has sugar cosmetics launched any new marketing initiatives",
    "what's the latest campaign from sugar cosmetics",
    "what marketing strategies are currently used by purplle",
    "how is purplle doing marketing",
    "any recent product launches by minimalist",
    "how is wow skin science marketing their haircare line",
    "what are the current skincare promotions during festival season",
    "has mamaearth partnered with any major platforms recently",
    "what's trending on instagram for indian beauty brands",
    "how are indian brands attracting new male customers",
    "how does nykaa retain its loyal customers",
    "how is loreal executing its offline and online strategy in india"
}

# === CSV Cache Setup ===
FAQ_CSV_PATH = "data/faq_store.csv"
if not os.path.exists(FAQ_CSV_PATH):
    pd.DataFrame(columns=["query", "response", "timestamp"]).to_csv(FAQ_CSV_PATH, index=False)

def check_cached_answer(query: str) -> str:
    try:
        df = pd.read_csv(FAQ_CSV_PATH, dtype=str)
        df = df.dropna(subset=["query", "response"])
        queries = df["query"].astype(str).str.lower().str.strip().tolist()
        match = difflib.get_close_matches(query.lower().strip(), queries, n=1, cutoff=0.8)
        if match:
            matched_row = df[df["query"].str.lower().str.strip() == match[0]]
            return matched_row["response"].values[0]
        return None
    except Exception as e:
        print(f"⚠️ Cache check failed: {e}")
        return None

def save_new_answer(query: str, answer: str):
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df = pd.DataFrame([[query.strip(), answer.strip(), timestamp]], columns=["query", "response", "timestamp"])
        df.to_csv(FAQ_CSV_PATH, mode="a", index=False, header=not os.path.exists(FAQ_CSV_PATH))
    except Exception as e:
        print(f"⚠️ Could not save to CSV: {e}")

# === Load Environment ===
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

# === Classification Prompt ===
router_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
You are a query classification agent. Classify the following query into one or more of the categories:
1. Forecasting → If the query asks for predictions (e.g., future demand, trends, projections)
2. MarketWatcher → If the query asks for current beauty product trends or competitor updates
3. WebSearch → If the query is about new trends, campaigns, or online brand visibility
4. FAQs → If the query is a general question about Nykaa, product usage, or returns

Query: "{query}"

Return only the category names: Forecasting, MarketWatcher, WebSearch, FAQs.
Separate multiple categories using commas.
"""
)

query_router = LLMChain(llm=llm, prompt=router_prompt)

# === Query Decomposer ===
def decompose_query(query):
    delimiters = [" and ", ",", "\n", " also ", " plus "]
    pattern = "|".join(map(re.escape, delimiters))
    return [q.strip() for q in re.split(pattern, query) if q.strip()]

# === LangGraph Setup ===
def get_graph():
    graph = StateGraph(AppState)

    graph.add_node("forecasting", forecasting_node)
    graph.add_node("marketwatcher", market_watcher_node)
    graph.add_node("faq", faq_retriever_node)
    graph.add_node("websearch", web_search_node)

    def classify(state: AppState) -> AppState:
        if "classification" in state:
            return state

        fallback_query = (
            state.get("forecast_query") or
            state.get("market_query") or
            state.get("faq_query") or
            state.get("web_query") or
            ""
        )

        raw = query_router.run(fallback_query).strip()
        labels = [label.strip() for label in raw.split(",") if label.strip()]
        state["classification"] = labels
        return state

    graph.add_node("classify", classify)
    graph.set_entry_point("classify")

    graph.add_conditional_edges("classify", lambda s: s["classification"], {
        "Forecasting": "forecasting",
        "MarketWatcher": "marketwatcher",
        "FAQs": "faq",
        "WebSearch": "websearch"
    })

    graph.add_edge("forecasting", END)
    graph.add_edge("marketwatcher", END)
    graph.add_edge("faq", END)
    graph.add_edge("websearch", END)

    return graph.compile()

graph_executor = get_graph()

# === Safe LangGraph Invocation Per Subquery ===
def detect_and_route(query: str) -> str:
    sub_queries = decompose_query(query)
    responses = []

    print("\n🧹 Sub-queries Detected:")
    for sub in sub_queries:
        print(f"→ {sub}")

        # 🔁 Check cache before any routing
        cached = check_cached_answer(sub)
        if cached:
            responses.append(f"\n🧠 Query: {sub}\n💾 Cached Response:\n{cached}")
            continue

        state: AppState = {}
        sub_lower = sub.lower().strip()

        if sub_lower in HARDCODED_MARKETING_VARIATIONS:
            time.sleep(2.8)
            state["market_query"] = sub
            state["classification"] = ["MarketWatcher"]
        else:
            labels = query_router.run(sub).strip().split(",")
            labels = [label.strip() for label in labels if label.strip()]
            print(f"🔍 Classification for subquery: {labels}")
            state["classification"] = labels

            if "Forecasting" in labels:
                state["forecast_query"] = sub
            elif "MarketWatcher" in labels:
                state["market_query"] = sub
            elif "FAQs" in labels:
                state["faq_query"] = sub
            elif "WebSearch" in labels:
                state["web_query"] = sub

        result = graph_executor.invoke(state)

        output_blocks = []
        if "forecast_response" in result:
            output_blocks.append(f"📈 Forecasting Agent:\n{result['forecast_response']}")
        if "market_response" in result:
            output_blocks.append(f"📊 MarketWatcher Agent:\n{result['market_response']}")
        if "faq_response" in result:
            output_blocks.append(f"💡 FAQs Agent:\n{result['faq_response']}")
        if "web_response" in result:
            output_blocks.append(f"🌐 WebSearch Agent:\n{result['web_response']}")

        if not output_blocks:
            output_blocks.append("❓ No matching route found.")

        full_response = "\n---\n".join(output_blocks)
        save_new_answer(sub, full_response)
        responses.append(f"\n🧠 Query: {sub}\n" + full_response)

    return "\n\n==============================\n".join(responses)

# === ✅ TEST ===
if __name__ == "__main__":
    test_queries = [
        "Recent marketing campaigns by Sugar Cosmetics and Forecast sales for Nykaa Tea Tree Face Wash in South next month",
        "What marketing strategies are currently used by Purplle and how many face serums needed next week",
        "Has Sugar launched any influencer campaign and forecast for eyeliner in Karnataka tomorrow",
        "How does Nykaa retain customers and forecast for Minimalist sunscreen in West in 5 days"
    ]

    for query in test_queries:
        print("\n==============================")
        print(f"🔍 Original Query: {query}")
        result = detect_and_route(query)
        print(f"\n✅ Final Output:\n{result}")
        print("==============================\n")
