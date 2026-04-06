import sqlite3
from typing import TypedDict, Annotated
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

# --- 1. Model Configuration ---
# temperature=0 ensures the model stays focused and doesn't hallucinate 
# or write long, unrelated essays (the Dr. Alex issue).
llm = ChatOllama(model="phi3", temperature=0)

# --- 2. State Definition ---
class ChatState(TypedDict):
    # Annotated with add_messages tells LangGraph to automatically 
    # append new messages to the history list.
    messages: Annotated[list[BaseMessage], add_messages]

# --- 3. The Logic Node ---
def chat_node(state: ChatState):
    """
    Processes the conversation. It checks for a System Message and 
    calls the Ollama model.
    """
    messages = state['messages']
    
    # Inject a System Message at the beginning if it doesn't exist.
    # This keeps Phi-3 focused on being a concise assistant.
    if not any(isinstance(m, SystemMessage) for m in messages):
        system_prompt = SystemMessage(content="You are a helpful and concise AI assistant.")
        messages = [system_prompt] + messages
    
    response = llm.invoke(messages)
    
    # Return only the new message; add_messages will merge it into the state.
    return {"messages": [response]}

# --- 4. Database & Persistence Setup ---
# We create a persistent SQLite file named 'chatbot_database.db'.
DB_PATH = "chatbot_database.db"

# CRITICAL: check_same_thread=False is necessary because Streamlit 
# runs on multiple threads, and SQLite needs to allow cross-thread access.
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
memory = SqliteSaver(conn)

# --- 5. Graph Construction ---
workflow = StateGraph(ChatState)

# Add the 'chat' node and define the workflow path
workflow.add_node("chat", chat_node)
workflow.add_edge(START, "chat")
workflow.add_edge("chat", END)

# Compile the chatbot with the SQLite checkpointer.
# This object is what the frontend imports.
chatbot = workflow.compile(checkpointer=memory)

# --- 6. Frontend Helper Functions ---

def retrieve_all_threads():
    """
    Scans the database and returns a list of all unique thread IDs.
    Used to populate the 'My Conversations' sidebar in Streamlit.
    """
    # .list(None) retrieves every checkpoint stored in the DB.
    all_checkpoints = memory.list(None)
    unique_threads = set()
    
    for checkpoint in all_checkpoints:
        tid = checkpoint.config['configurable'].get('thread_id')
        if tid:
            unique_threads.add(tid)
            
    return list(unique_threads)

def get_chat_history(thread_id: str):
    """
    Retrieves and formats the message history for a specific thread_id.
    Converts LangChain message objects into a list of dictionaries for Streamlit.
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    # Fetch current state values for this thread
    state = chatbot.get_state(config)
    messages = state.values.get("messages", [])
    
    formatted_history = []
    for msg in messages:
        # Convert message types to simple string roles for the UI
        if isinstance(msg, HumanMessage):
            formatted_history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            formatted_history.append({"role": "assistant", "content": msg.content})
            
    return formatted_history