import os
import streamlit as st
from dotenv import load_dotenv
from router.smart_query_router import detect_and_route
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# === Fix Streamlit + Torch Path Issues ===
os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"

# === Load API Keys ===
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# === Gemini Small Talk Enhancer ===
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.6)
smalltalk_prompt = PromptTemplate.from_template("""
You are a professional assistant working for Nykaa's Business Intelligence chatbot.

Your task is to rewrite vague, short, or conversational queries into clear, structured business analysis queries.

Guidelines:
- Preserve technical/business intent, don't change data points or product names.
- For vague queries like “how’s this product doing?”, rewrite to be region- and product-specific if possible.
- For casual phrasing like “can you tell me about...”, reframe it formally.
- Do NOT rewrite queries that already seem structured or detailed.

Example:
Input: "how are things looking for blush?"
Output: "Provide a demand forecast for blush products across India for the upcoming month."

Now, rewrite this user query:
Query: {query}

Return only the rewritten query:
""")
rephraser_chain = smalltalk_prompt | llm

def rephrase_input(query):
    try:
        return rephraser_chain.invoke({"query": query}).content.strip()
    except:
        return query

# === Streamlit UI Config ===
st.set_page_config(page_title="Nykaa BI Assistant", layout="wide")

# === Drop Shadow Style for Images ===
st.markdown("""
    <style>
        img {
            border-radius: 16px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# === Centered and Styled Logo Using st.image() ===
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    #st.image("assets/nykaa_logo.png", width=320)

# === Title and Product Highlights ===
st.markdown("""
    <div style='text-align: center;'>
        <h1 style='margin-bottom: 5px; background: linear-gradient(to right, #d4145a, #fbb03b); -webkit-background-clip: text; color: transparent; font-size: 2.6em;'>
            Nykaa Business Intelligence Assistant
        </h1>
        <p style='font-size: 16px; color: gray; margin-top: -10px;'>Focus Products: 💄 Lipstick · 💅 Blush · 💧 Serum · 🧴 Foundation · 👁 Eyeliner</p>
    </div>
""", unsafe_allow_html=True)

# === Introduction ===
st.markdown("""
This assistant supports Nykaa's business stakeholders with actionable insights related to:
- 📈 **Demand Forecasting** (regional product-level projections)
- 📊 **Market & Competitor Intelligence** (product comparisons, pricing, performance)
- 🌐 **Web Intelligence** (online trends, brand campaigns, emerging categories)
- 💡 **Customer FAQs** (policy, delivery, returns)
""")

# === Query Input ===
query = st.text_area(
    "Enter your business query:",
    height=100,
    placeholder="e.g., Project sales for face serum in South India next month and tell me about key skincare launches by Purplle.",
)

@st.cache_data(ttl=3600)
def get_response(user_query):
    return detect_and_route(user_query)

if st.button("Run Analysis") and query.strip():
    with st.spinner("Analyzing your query. Please wait..."):
        friendly_query = rephrase_input(query)
        result = get_response(friendly_query)
        st.markdown("---")

        for block in result.split("=============================="):
            block = block.strip()
            if not block:
                continue
            lines = block.split("\n")
            st.markdown("<div style='background-color: #f9f9f9; padding: 20px; border-radius: 10px; border: 1px solid #e1e1e1;'>", unsafe_allow_html=True)

            for line in lines:
                if line.startswith("🧠 Query"):
                    st.markdown(f"<h4 style='color:#1f77b4'>{line}</h4>", unsafe_allow_html=True)
                elif line.startswith("📈 Forecasting Agent"):
                    st.markdown(f"<strong style='color:#2ecc71'>{line}</strong>", unsafe_allow_html=True)
                elif line.startswith("📊 MarketWatcher Agent"):
                    st.markdown(f"<strong style='color:#9b59b6'>{line}</strong>", unsafe_allow_html=True)
                elif line.startswith("🌐 WebSearch Agent"):
                    st.markdown(f"<strong style='color:#3498db'>{line}</strong>", unsafe_allow_html=True)
                elif line.startswith("💡 FAQs Agent"):
                    st.markdown(f"<strong style='color:#f39c12'>{line}</strong>", unsafe_allow_html=True)
                elif line.startswith(("🔮", "📦", "📅", "🗺️", "📌", "📊")):
                    st.markdown(f"<div style='margin-left:10px; color:#2c3e50;'>{line}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='margin-left:20px; color:#7f8c8d;'>{line}</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<h3 style='color:#e74c3c;'>🔍 Final Business Insights</h3>", unsafe_allow_html=True)
        st.markdown(f"""
            <div style='background-color: #ecf0f1; padding: 20px; border-radius: 10px; color: #34495e;'>
                <strong>{result}</strong>
            </div>
        """, unsafe_allow_html=True)

# === Footer ===
st.markdown("""
<hr style='margin-top: 50px;'/>
<div style='text-align:center; font-size:13px; color:gray;'>
    Built with 💖 for Nykaa’s Strategy & Intelligence Team
</div>
""", unsafe_allow_html=True)
