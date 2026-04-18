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
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import requests

load_dotenv()

# 1. Models
llm = ChatOllama(model="phi3", temperature=0)
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# 2. Per-thread Memory (In-process cache for retrievers)
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}

def _get_retriever(thread_id: Optional[str]):
    tid = str(thread_id) if thread_id else None
    return _THREAD_RETRIEVERS.get(tid)

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    if not file_bytes:
        raise ValueError("No bytes received.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})

        tid = str(thread_id)
        _THREAD_RETRIEVERS[tid] = retriever
        _THREAD_METADATA[tid] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
        return _THREAD_METADATA[tid]
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# 3. Tools
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform basic arithmetic (add, sub, mul, div)."""
    try:
        if operation == "add": res = first_num + second_num
        elif operation == "sub": res = first_num - second_num
        elif operation == "mul": res = first_num * second_num
        elif operation == "div": res = first_num / second_num if second_num != 0 else "Error: Div by Zero"
        else: return {"error": "Invalid op"}
        return {"result": res}
    except Exception as e: return {"error": str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch latest stock price for a symbol (e.g. AAPL)."""
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    return requests.get(url).json()

@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """Search the PDF for this thread. Include thread_id."""
    retriever = _get_retriever(thread_id)
    if not retriever:
        return {"error": "No document indexed for this thread. Ask user to upload one."}
    
    result = retriever.invoke(query)
    return {
        "context": [d.page_content for d in result],
        "source": _THREAD_METADATA.get(str(thread_id), {}).get("filename")
    }

tools = [search_tool, get_stock_price, calculator, rag_tool]
llm_with_tools = llm.bind_tools(tools)

# 4. Graph Logic
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState, config=None):
    thread_id = config.get("configurable", {}).get("thread_id") if config else "unknown"
    sys_msg = SystemMessage(content=f"You are a local AI. Use 'rag_tool' for PDF questions (thread_id: {thread_id}).")
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"], config=config)]}

conn = sqlite3.connect(database="chatbot_local.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

builder = StateGraph(ChatState)
builder.add_node("chat_node", chat_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "chat_node")
builder.add_conditional_edges("chat_node", tools_condition)
builder.add_edge("tools", "chat_node")
chatbot = builder.compile(checkpointer=checkpointer)

# 5. NEW HELPER FUNCTIONS (Essential for Frontend)
def retrieve_all_threads():
    try:
        return list({c.config["configurable"]["thread_id"] for c in checkpointer.list(None)})
    except: return []

def thread_has_document(thread_id: str) -> bool:
    """Checks if a retriever exists for the current session."""
    return str(thread_id) in _THREAD_RETRIEVERS

def thread_document_metadata(thread_id: str) -> dict:
    """Returns the name and stats of the uploaded document."""
    return _THREAD_METADATA.get(str(thread_id), {})