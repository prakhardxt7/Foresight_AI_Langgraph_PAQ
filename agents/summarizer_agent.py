import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

# Create PromptTemplate for Summarizing Agent Responses
summary_prompt = PromptTemplate.from_template("""
You are a senior business strategist working in a retail analytics team. Your task is to synthesize the responses from different agents (Forecasting, MarketWatcher, WebSearch, FAQs) and generate a summarized business insight.

The following responses need to be summarized:

1. **Forecasting Response**: {forecasting_response}
2. **MarketWatcher Response**: {marketwatcher_response}
3. **WebSearch Response**: {websearch_response}
4. **FAQ Response**: {faq_response}

In your summary, provide:
1. A **comprehensive business insight** based on the combined data.
2. Key **recommendations for the business** (inventory management, marketing, pricing, etc.)
3. Identify any **gaps in data** and provide actionable next steps to fill them.

Format the response as follows:

📌 **Business Summary:**
- [Insight based on combined data]
- [Key recommendations]

📊 **Next Steps:**
- [Actionable next steps]

**Note**: Avoid errors and use the most relevant data for actionable insights.
""")

# Create the LLMChain for the summarizer
summarizer_chain = LLMChain(llm=llm, prompt=summary_prompt)

class SummarizerAgent:
    def __init__(self):
        pass

    def generate_summary(self, forecasting_response, marketwatcher_response, websearch_response, faq_response):
        # Combine all agent responses into the summary prompt
        try:
            summary = summarizer_chain.run({
                "forecasting_response": forecasting_response,
                "marketwatcher_response": marketwatcher_response,
                "websearch_response": websearch_response,
                "faq_response": faq_response
            })
            return summary.strip()
        except Exception as e:
            return f"❌ Error in generating summary: {str(e)}"

# Test the Summarizer Agent
if __name__ == "__main__":
    summarizer = SummarizerAgent()

    # Example responses from agents (you would get these from actual agents in practice)
    forecasting_response = "The demand for Nykaa SKINRX Vitamin C Serum is forecasted to be 500 units in the next quarter."
    marketwatcher_response = "Purplle has launched a Vitamin C serum at a 15% lower price point, posing a potential competitive threat."
    websearch_response = "The latest trends indicate a growing interest in natural skincare products, with many new brands emerging."
    faq_response = "The customer has asked about return policies. The standard return process is 7 days from the delivery date."

    # Generate the summary based on agent responses
    final_summary = summarizer.generate_summary(
        forecasting_response,
        marketwatcher_response,
        websearch_response,
        faq_response
    )

    print("\n🔑 Final Business Insight Summary:")
    print(final_summary)
