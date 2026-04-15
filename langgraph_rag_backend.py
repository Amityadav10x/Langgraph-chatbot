from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
# --- NEW OLLAMA IMPORTS ---
from langchain_ollama import ChatOllama, OllamaEmbeddings
# --------------------------
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import requests

load_dotenv()

# -------------------
# 1. Local LLM + Local Embeddings
# -------------------
# Ensure you have run: ollama pull phi3
llm = ChatOllama(model="phi3", temperature=0)

# Ensure you have run: ollama pull mxbai-embed-large
# 'mxbai-embed-large' is currently top-tier for local RAG
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# -------------------
# 2. PDF retriever store (per thread)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}

def _get_retriever(thread_id: Optional[str]):
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    if not file_bytes:
        raise ValueError("No bytes received.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        # Using Local Ollama Embeddings here
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return _THREAD_METADATA[str(thread_id)]
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

# -------------------
# 3. Tools (Same logic, bound to local LLM)
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform a basic arithmetic operation (add, sub, mul, div)."""
    if operation == "add": result = first_num + second_num
    elif operation == "sub": result = first_num - second_num
    elif operation == "mul": result = first_num * second_num
    elif operation == "div": result = first_num / second_num if second_num != 0 else "Error: Div by Zero"
    else: return {"error": "Unsupported operation"}
    return {"result": result}

@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch latest stock price for a symbol (e.g. AAPL)."""
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    return requests.get(url).json()

@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve info from the thread's PDF. 
    ALWAYS include the thread_id provided in your instructions.
    """
    retriever = _get_retriever(thread_id)
    if not retriever:
        return {"error": "No document uploaded for this session."}

    result = retriever.invoke(query)
    return {
        "context": [d.page_content for d in result],
        "source": _THREAD_METADATA.get(str(thread_id), {}).get("filename")
    }

tools = [search_tool, get_stock_price, calculator, rag_tool]
# Bind tools to Phi-3
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 4. State & Nodes
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState, config=None):
    thread_id = config.get("configurable", {}).get("thread_id") if config else "unknown"

    system_message = SystemMessage(
        content=(
            f"You are a local AI Assistant. Use 'rag_tool' for PDF questions with thread_id: {thread_id}. "
            "Be concise. If you need info you don't have, check the web or the PDF."
        )
    )

    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}

# -------------------
# 5. Graph Assembly
# -------------------
conn = sqlite3.connect(database="chatbot_local.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

builder = StateGraph(ChatState)
builder.add_node("chat_node", chat_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "chat_node")
builder.add_conditional_edges("chat_node", tools_condition)
builder.add_edge("tools", "chat_node")

chatbot = builder.compile(checkpointer=checkpointer)