import os
import sqlite3
from typing import TypedDict, Annotated
from dotenv import load_dotenv

# --- LOAD ENV ---
load_dotenv()

# --- LANGSMITH CONFIG ---
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Carrer_OS_Chat"

# --- DEBUG LANGSMITH CONNECTION ---
from langsmith import Client
try:
    client = Client()
    print("✅ LangSmith Connected:", list(client.list_projects()))
except Exception as e:
    print("❌ LangSmith Connection Failed:", e)

# --- IMPORTS ---
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

# --- MODEL ---
llm = ChatOllama(
    model="phi3",
    temperature=0,
    streaming=True
)

# --- STATE ---
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# --- NODE ---
def chat_node(state: ChatState):
    try:
        messages = state["messages"]

        # Inject system prompt
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [
                SystemMessage(content="You are a helpful and concise AI assistant.")
            ] + messages

        print("📩 Messages:", messages)

        response = llm.invoke(messages)

        return {"messages": [response]}

    except Exception as e:
        print("❌ NODE ERROR:", e)
        raise e

# --- DATABASE ---
DB_PATH = "chatbot_database.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
memory = SqliteSaver(conn)

# --- GRAPH ---
workflow = StateGraph(ChatState)
workflow.add_node("chat", chat_node)
workflow.add_edge(START, "chat")
workflow.add_edge("chat", END)

chatbot = workflow.compile(checkpointer=memory)

# --- 🔥 LANGSMITH WARMUP ---
try:
    chatbot.invoke(
        {"messages": [HumanMessage(content="init")]},
        config={"configurable": {"thread_id": "debug"}}
    )
    print("🔥 LangSmith Warmup Success")
except Exception as e:
    print("Warmup Failed:", e)

# --- HELPERS ---
def retrieve_all_threads():
    all_checkpoints = memory.list(None)
    unique_threads = set()

    for checkpoint in all_checkpoints:
        tid = checkpoint.config["configurable"].get("thread_id")
        if tid:
            unique_threads.add(tid)

    return list(unique_threads)


def get_chat_history(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}

    state = chatbot.get_state(config)
    messages = state.values.get("messages", [])

    formatted_history = []

    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted_history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            formatted_history.append({"role": "assistant", "content": msg.content})

    return formatted_history
