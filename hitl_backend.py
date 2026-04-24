from typing import Annotated, TypedDict, Union
import sqlite3
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

# --- 1. Setup ollama Phi-3 ---
llm = ChatOllama(model="phi3", temperature=0)

@tool
def basic_calculator(expression: str):
    """Evaluates a mathematical expression (e.g., '2 + 2')."""
    try:
        return {"result": eval(expression)}
    except Exception as e:
        return {"error": str(e)}

tools = [basic_calculator]
llm_with_tools = llm.bind_tools(tools)

# --- 2. State & Nodes ---
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def assistant_node(state: State):
    # Standard LLM call
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# --- 3. Building graph ---
builder = StateGraph(State)
builder.add_node("assistant", assistant_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

# --- 4. The HITL data base connection---
conn = sqlite3.connect("hitl_storage.db", check_same_thread=False)
memory = SqliteSaver(conn)

# This tells LangGraph to PAUSE before the tools node runs
chatbot = builder.compile(checkpointer=memory, interrupt_before=["tools"])
