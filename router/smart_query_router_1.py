import os
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from agents.faq_retriever import retrieve_faq_answer
from agents.forecasting_agent_v1 import forecast_demand

# Load environment variable
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

# Classification Prompt
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
query_router = LLMChain(llm=llm, prompt=router_prompt)

def classify_query(query):
    return query_router.run(query).strip()

# Decompose multi-intent queries
def decompose_query(query):
    delimiters = [" and ", ",", "\n", " also ", " plus "]
    pattern = "|".join(map(re.escape, delimiters))
    return [q.strip() for q in re.split(pattern, query) if q.strip()]

# Route each sub-query to its appropriate agent
def route_and_answer(query):
    classification = classify_query(query)

    if "FAQs" in classification:
        return retrieve_faq_answer(query)
    elif "Forecasting" in classification:
        return forecast_demand(query)
    elif "MarketWatcher" in classification:
        return "[MarketWatcher Agent] Placeholder response for: " + query
    else:
        return "❓ Could not classify this query."

# Main Agent Detector with Routing
def detect_and_route(query):
    sub_queries = decompose_query(query)
    responses = []

    print(f"\n🧹 Sub-queries Detected:")
    for q in sub_queries:
        classification = classify_query(q)
        print(f'🔹 "{q}" → {classification}')
        result = route_and_answer(q)
        responses.append(f"\n➡️ {q}\n{result}")
    return "\n".join(responses)

# Test queries
if __name__ == "__main__":
    test_queries = [
        "How do I return a product I ordered on Nykaa and get my money back?",
        "Predict the demand for lip gloss in West India for the next quarter.",
        "Tell me the top trending face creams among Gen Z in India right now.",
        "Forecast serum demand in Delhi next month and tell me what’s trending in eye makeup.",
        "uhh how get refund nykaa? nd serum best now?",
        "people buying what lipstick these days? also can i change order?",
        "Give me insights into beauty product preferences in India."
    ]

    for query in test_queries:
        print("\n==============================")
        print(f"🔍 Original Query: {query}")
        full_response = detect_and_route(query)
        print(f"\n✅ Final Response:{full_response}")
        print("==============================\n")
