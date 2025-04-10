# File: agents/web_search_agent.py

import os
from dotenv import load_dotenv
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun

# Load API keys
load_dotenv()

# === Gemini LLM ===
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,
    convert_system_message_to_human=True
)

# === Web Search Tool ===
search = DuckDuckGoSearchRun()

# === Tool Config ===
tools = [
    Tool(
        name="DuckDuckGo Search",
        func=search.run,
        description="Useful for finding competitor products, market trends, campaigns, and brand launches."
    )
]

# === Initialize Agent ===
web_agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

def run_web_search(query: str) -> str:
    """Run a web search for any market insight or competitor-related query."""
    response = web_agent.run(query)
    return response


# === ✅ Test ===
if __name__ == "__main__":
    test_queries = [
        "What are some trending face serums in India?",
        "Top competitors for Nykaa Lip Balm",
        "New launches by Purplle in skincare",
        "Recent marketing campaigns by Sugar Cosmetics"
    ]

    for i, q in enumerate(test_queries, 1):
        print(f"\n🔍 Web Query {i}: {q}")
        print(run_web_search(q))
