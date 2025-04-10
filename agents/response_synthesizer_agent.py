import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# === Initialize Gemini LLM ===
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

# === Synthesis Prompt ===
synthesis_prompt = PromptTemplate.from_template("""
You are an intelligent business assistant for Nykaa's strategic and operations team.

You have received the following inputs from different agents. Synthesize these responses and provide a comprehensive, actionable business recommendation.

- 📈 **Forecasting**: {forecasting_response}
- 📊 **MarketWatcher**: {marketwatcher_response}
- 🌐 **WebSearch**: {websearch_response}
- 💡 **FAQ**: {faq_response}

Focus on the following:
- Combine insights from each agent and identify business opportunities.
- Provide strategic recommendations based on the data provided by each agent.
- Suggest inventory planning, marketing strategies, and pricing ideas, as well as competitor threats or trends.

Return the final synthesis in a professional business report format.
""")

class ResponseSynthesizerAgent:
    def __init__(self):
        pass

    def synthesize_responses(self, forecasting_response, marketwatcher_response, websearch_response, faq_response):
        # Prepare the input data to synthesize responses from multiple agents
        input_data = synthesis_prompt.format(
            forecasting_response=forecasting_response,
            marketwatcher_response=marketwatcher_response,
            websearch_response=websearch_response,
            faq_response=faq_response
        )

        # Pass the input data to the LLM for synthesis
        try:
            final_response = llm.invoke(input_data).content.strip()
        except Exception as e:
            final_response = f"❌ Error synthesizing responses: {str(e)}"
        
        return final_response

# Test the Response Synthesizer Agent
if __name__ == "__main__":
    synthesizer_agent = ResponseSynthesizerAgent()

    # Example responses from different agents
    forecasting_response = "The expected demand for Nykaa SKINRX Vitamin C Serum is 500 units in the next quarter."
    marketwatcher_response = "Purplle has launched a similar Vitamin C serum with a 15% lower price point, posing a potential threat."
    websearch_response = "Trending face serums include brands like The Ordinary and L'Oréal, gaining popularity in India."
    faq_response = "The customer inquiry related to return policies has been resolved with standard instructions."

    # Synthesize all responses into one actionable business report
    final_synthesis = synthesizer_agent.synthesize_responses(
        forecasting_response, marketwatcher_response, websearch_response, faq_response
    )

    print("\n📊 Final Business Insight:\n")
    print(final_synthesis)
