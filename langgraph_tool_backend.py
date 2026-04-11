import os
import sqlite3
import requests
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

# --- LOAD ENV ---
load_dotenv()

# --- MODEL ---
llm = ChatOllama(
    model="phi3",   # ✅ tool-supported model
    temperature=0,
    streaming=True
)

# --- TOOL 1: SEARCH ---
search_tool = DuckDuckGoSearchRun()

# --- TOOL 2: CALCULATOR ---
@tool(description="Perform arithmetic operations: add, sub, mul, div")
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    try:
        if operation == "add":
            return {"result": first_num + second_num}
        elif operation == "sub":
            return {"result": first_num - second_num}
        elif operation == "mul":
            return {"result": first_num * second_num}
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero"}
            return {"result": first_num / second_num}
        else:
            return {"error": "Invalid operation"}
    except Exception as e:
        return {"error": str(e)}

# --- TOOL 3: STOCK ---
@tool(description="Get stock price for a company symbol like AAPL, TSLA")
def get_stock_price(symbol: str) -> dict:
    API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

    if not API_KEY:
        return {"error": "API key missing"}

    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={API_KEY}"

    try:
        data = requests.get(url).json()
        price = data.get("Global Quote", {}).get("05. price")

        if not price:
            return {"error": "Invalid symbol"}

        return {"symbol": symbol, "price": price}
    except Exception as e:
        return {"error": str(e)}

# --- TOOLS ---
tools = [get_stock_price, search_tool, calculator]

# --- STATE ---
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# --- SMART TOOL DECISION FUNCTION ---
def should_use_tools(user_input: str) -> bool:
    keywords = [
        "calculate", "sum", "add", "multiply", "divide",
        "stock", "price", "search", "latest", "news"
    ]
    return any(word in user_input.lower() for word in keywords)

# --- NODE ---
def chat_node(state: ChatState):
    try:
        messages = state["messages"]
        user_input = messages[-1].content.lower()

        # SYSTEM PROMPT
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [
                SystemMessage(content="""
You are a helpful AI assistant.

Rules:
- Respond normally for greetings like hi, hello
- Use tools ONLY when needed
- Do NOT call tools unnecessarily
""")
            ] + messages

        print("🧠 INPUT:", user_input)

        # ✅ SMART ROUTING
        if should_use_tools(user_input):
            response = llm.bind_tools(tools).invoke(messages)
        else:
            response = llm.invoke(messages)

        print("🤖 OUTPUT:", response)

        return {"messages": [response]}

    except Exception as e:
        print("❌ ERROR:", e)
        return {"messages": [AIMessage(content="Something went wrong.")]}

# --- TOOL NODE ---
tool_node = ToolNode(tools)

# --- DATABASE ---
conn = sqlite3.connect("chatbot_database.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

# --- GRAPH ---
graph = StateGraph(ChatState)

graph.add_node("chat", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat")
graph.add_conditional_edges("chat", tools_condition)
graph.add_edge("tools", "chat")
graph.add_edge("chat", END)

# --- COMPILE ---
chatbot = graph.compile(checkpointer=checkpointer)

# --- HELPERS ---
def retrieve_all_threads():
    threads = set()
    for checkpoint in checkpointer.list(None):
        tid = checkpoint.config["configurable"].get("thread_id")
        if tid:
            threads.add(tid)
    return list(threads)

def get_chat_history(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    state = chatbot.get_state(config)
    messages = state.values.get("messages", [])

    formatted = []
    for msg in messages:
        if msg.type == "human":
            formatted.append({"role": "user", "content": msg.content})
        elif msg.type == "ai":
            formatted.append({"role": "assistant", "content": msg.content})

    return formatted