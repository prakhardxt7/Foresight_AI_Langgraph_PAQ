from dotenv import load_dotenv
import os
from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Load API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini Flash 1.5
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    temperature=0.3,
    google_api_key=GOOGLE_API_KEY
)

# Define state (we track user message and node outputs)
class ChatState(dict):
    pass

# STEP 1: Classify input
def classify_node(state: ChatState):
    user_input = state.get("user_input", "")
    print(f"\n🔍 Classifying: {user_input}")
    
    # Simple classification logic using keywords
    if any(word in user_input.lower() for word in ["hi", "hello", "hey"]):
        return "greet"
    elif any(word in user_input.lower() for word in ["bye", "goodbye", "see you"]):
        return "farewell"
    else:
        return "default"

# STEP 2: Respond to greeting
def greet_node(state: ChatState):
    msg = HumanMessage(content="Greet the user in a cheerful way.")
    response = llm.invoke([msg])
    print(f"🤖 Greet: {response.content}")
    return {"response": response.content}

# STEP 3: Respond to farewell
def farewell_node(state: ChatState):
    msg = HumanMessage(content="Say goodbye in a kind and respectful tone.")
    response = llm.invoke([msg])
    print(f"🤖 Farewell: {response.content}")
    return {"response": response.content}

# STEP 4: Fallback for unknown inputs
def default_node(state: ChatState):
    msg = HumanMessage(content=f"Reply to the user query: '{state['user_input']}' politely, saying you don't understand.")
    response = llm.invoke([msg])
    print(f"🤖 Default: {response.content}")
    return {"response": response.content}

# Build the graph
builder = StateGraph(ChatState)

# Add nodes
builder.add_node("classify", classify_node)
builder.add_node("greet", greet_node)
builder.add_node("farewell", farewell_node)
builder.add_node("default", default_node)

# Set entry point
builder.set_entry_point("classify")

# Add conditional edges
builder.add_conditional_edges(
    "classify",
    lambda state: classify_node(state),  # classifier returns the next node name
    {
        "greet": "greet",
        "farewell": "farewell",
        "default": "default"
    }
)

# All paths lead to END
builder.add_edge("greet", END)
builder.add_edge("farewell", END)
builder.add_edge("default", END)

# Compile the graph
graph = builder.compile()

# Run it interactively
if __name__ == "__main__":
    while True:
        user_input = input("\n🧑 You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        state = {"user_input": user_input}
        final = graph.invoke(state)
        print(f"\n✅ Final Response: {final.get('response')}")
