# web_search_agent.py

import os
from dotenv import load_dotenv
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.prompts import PromptTemplate

# === Load environment variables ===
load_dotenv()

# === Gemini Setup ===
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,
    convert_system_message_to_human=True
)

# === DuckDuckGo Tool ===
search_tool = DuckDuckGoSearchRun()

tools = [
    Tool(
        name="DuckDuckGo Search",
        func=search_tool.run,
        description="Useful for searching online competitor data, beauty trends, or new launches"
    )
]

# === Agent Setup ===
web_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False
)

# === Gemini Summary Prompt ===
summary_prompt = PromptTemplate.from_template("""
You are a senior market analyst assistant for a retail beauty brand.
Your task is to extract actionable intelligence and competitive insights from web search content.

Search Result:
<<START_RESULT>>
{search_result}
<<END_RESULT>>

Focus areas:
- Identify trending products and active brands in the beauty industry
- Extract competitor mentions (e.g., Nykaa, Purplle, Sugar Cosmetics) and their current market activities
- Summarize any new campaigns, influencers, or marketing pushes that could impact Nykaa
- Provide actionable recommendations based on trends and competitor insights
- Mention potential opportunities or threats for Nykaa

Your answer should focus on:
1. Highlighting the key trends in products or brands relevant to Nykaa's market.
2. Summarizing the most important competitors' actions and how Nykaa can respond.
3. Provide strategic advice, e.g., pricing adjustments, promotions, or product focus areas.
4. If there are no clear insights or trends, return "No concrete competitor/product insight found."

Format response as:
📌 **Key Insights:**
- [Actionable insight about trends or competitors]
- [Recommended strategy for Nykaa]

📊 **Competitors Mentioned:**
- [List of competitors or brands mentioned in the search result]
If no valid insight, return "No concrete competitor/product insight found."
""")

# === Main Web Agent Query Function ===
def query_web(query: str) -> str:
    try:
        raw_result = web_agent.run(query)
        prompt = summary_prompt.format(search_result=raw_result)
        summary = llm.invoke(prompt).content.strip()
        return summary
    except Exception as e:
        return f"❌ Web search failed: {str(e)}"


# === ✅ LangGraph-compatible Node ===
def web_search_node(state: dict) -> dict:
    query = state.get("web_query", "")
    summary = query_web(query)
    state["web_response"] = summary
    return state