import streamlit as st
import uuid
from hitl_backend import chatbot
from langchain_core.messages import HumanMessage
from hitl_backend import chatbot


st.set_page_config(page_title="Carrer_OS | HITL Lab", layout="centered")
st.title("🛡️ Human-in-the-Loop AI")

# Initialize Session
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

config = {"configurable": {"thread_id": st.session_state.thread_id}}

# --- 1. Check Graph State ---
snapshot = chatbot.get_state(config)
is_waiting = len(snapshot.next) > 0  # If 'next' is not empty, it's paused

# --- 2. Sidebar History ---
with st.sidebar:
    st.info(f"Thread: {st.session_state.thread_id}")
    if st.button("Clear Conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# --- 3. UI logic ---
# Display past messages
for msg in snapshot.values.get("messages", []):
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(msg.content if msg.content else "Checking tools...")

# --- 4. HITL Control Panel ---
if is_waiting:
    st.warning("🚨 **Human Review Required:** The AI wants to use a tool.")
    
    # Show what the AI is planning to do
    last_msg = snapshot.values["messages"][-1]
    if last_msg.tool_calls:
        st.code(f"Tool: {last_msg.tool_calls[0]['name']}\nArgs: {last_msg.tool_calls[0]['args']}")

    col1, col2 = st.columns(2)
    if col1.button("✅ Approve & Run"):
        # Passing None tells the graph to RESUME from the interrupt
        for event in chatbot.stream(None, config=config, stream_mode="values"):
            pass
        st.rerun()
        
    if col2.button("❌ Deny (Coming Soon)"):
        st.error("In a real app, you could edit the state here!")

# --- 5. Chat Input ---
if not is_waiting:
    if prompt := st.chat_input("Ask me to calculate something (e.g., 'what is 55 * 12?')"):
        # Start the graph
        for event in chatbot.stream({"messages": [HumanMessage(content=prompt)]}, config=config, stream_mode="values"):
            pass
        st.rerun()