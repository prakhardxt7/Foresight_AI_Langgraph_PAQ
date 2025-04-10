import streamlit as st
import os
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

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

def decompose_query(query):
    delimiters = [" and ", ",", "\n", " also ", " plus "]
    pattern = "|".join(map(re.escape, delimiters))
    return [q.strip() for q in re.split(pattern, query) if q.strip()]

def detect_agents_involved(query):
    sub_queries = decompose_query(query)
    involved_agents = set()

    results = []
    for q in sub_queries:
        classification = classify_query(q)
        results.append(f'🔹 "{q}" → {classification}')
        for agent in ["Forecasting", "MarketWatcher", "FAQs"]:
            if agent in classification:
                involved_agents.add(agent)

    return results, involved_agents

# Streamlit UI
st.title("Query Routing System")
st.write("Enter a query to classify and identify relevant agents.")

query = st.text_area("Enter your query:")

if st.button("Classify Query"):
    if query:
        results, agents = detect_agents_involved(query)
        st.subheader("Sub-query Classification:")
        for res in results:
            st.write(res)
        st.subheader("Agents Required:")
        st.write(", ".join(agents) if agents else "No relevant agents detected.")
    else:
        st.warning("Please enter a query before classifying.")
