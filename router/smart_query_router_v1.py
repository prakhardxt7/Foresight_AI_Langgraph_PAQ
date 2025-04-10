# File: router/smart_query_router.py

import sys
import os
import re
from dotenv import load_dotenv

# Add project root path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from agents.faq_retriever import retrieve_faq_answer
from agents.forecasting_agent import forecast_demand
from agents.market_watcher_agent import MarketWatcherAgent
from agents.web_search_agent import query_web

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

# PromptTemplate -> LLM
router_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
You are a smart query classification assistant.
Classify the following user query into one or more of these categories:
1. Forecasting → Predicting demand, units, or sales in future
2. MarketWatcher → Competitor trends, comparisons, product performance, or market insights
3. WebSearch → Missing competitor data, product trends, brand campaigns, news
4. FAQs → General questions about Nykaa, return/refund, delivery, product usage

Query: "{query}"

Return only the category names. Example: Forecasting, MarketWatcher
"""
)

# Composable chain
query_router = router_prompt | llm

# === Core Classification ===
def classify_query(query):
    return query_router.invoke({"query": query}).content.strip()

# === Decompose compound queries ===
def decompose_query(query):
    delimiters = [" and ", ",", "\n", " also ", " plus "]
    pattern = "|".join(map(re.escape, delimiters))
    return [q.strip() for q in re.split(pattern, query) if q.strip()]

# === Intelligent Router ===
def route_and_answer(query):
    classification = classify_query(query)
    print(f"\n🔍 Classification: {classification}")

    answers = []

    if "FAQs" in classification:
        answers.append("💡 FAQs Agent:\n" + retrieve_faq_answer(query))

    if "Forecasting" in classification:
        forecast_response = forecast_demand(query)
        answers.append("📈 Forecasting Agent:\n" + forecast_response)

    if "MarketWatcher" in classification:
        market_agent = MarketWatcherAgent()
        market_response = market_agent.compare_product(product_name=query, region="West")
        if isinstance(market_response, dict) and "error" not in market_response.get("status", ""):
            summary = market_response.get("summary") or str(market_response)
            answers.append("📊 MarketWatcher Agent:\n" + summary)
        else:
            message = market_response["message"] if isinstance(market_response, dict) else str(market_response)
            answers.append("📊 MarketWatcher Agent Error: " + message)

    if "WebSearch" in classification:
        web_response = query_web(query)
        answers.append("🌐 WebSearch Agent:\n" + web_response)

    if not answers:
        return "❓ Sorry, I couldn’t route this query. Try rephrasing."

    return "\n---\n".join(answers)

# === Master Orchestrator ===
def detect_and_route(query):
    sub_queries = decompose_query(query)
    responses = []

    print("\n🧹 Sub-queries Detected:")
    for sub in sub_queries:
        print(f"→ {sub}")
        res = route_and_answer(sub)
        responses.append(f"\n🧠 Query: {sub}\n{res}")

    return "\n\n==============================\n".join(responses)


# === ✅ TEST ===
if __name__ == "__main__":
    test_queries = [
        #"How do I return a product I ordered on Nykaa and get my money back?",
        #"Predict the demand for lip gloss in West India for the next quarter.",
        #"Tell me the top trending face creams among Gen Z in India right now.",
        #"Forecast serum demand in Delhi next month and tell me what’s trending in eye makeup.",
        #"uhh how get refund nykaa? nd serum best now?",
        #"people buying what lipstick these days? also can i change order?",
        #"Give me insights into beauty product preferences in India.",
        #"New launches by Purplle in skincare and forecast toner sales in South India"
        'Forecast Nykaa SKINRX Vitamin C Serum demand in South next month, tell me what’s trending in eye makeup, list new launches by Purplle in skincare, and how can I return a product on Nykaa?'
    ]

    for query in test_queries:
        print("\n==============================")
        print(f"🔍 Original Query: {query}")
        result = detect_and_route(query)
        print(f"\n✅ Final Output:\n{result}")
        print("==============================\n")
