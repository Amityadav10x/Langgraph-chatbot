import streamlit as st
import uuid
from langchain_core.messages import HumanMessage
from langgrpah_database_backend import chatbot, retrieve_all_threads, get_chat_history
# --- Page Config --
st.set_page_config(page_title="LangGraph Chat", layout="centered")
st.title("🤖 Local AI Chatbot")

# --- Initialize Session State ---
if 'all_threads' not in st.session_state:
    # Load old threads from DB immediately
    st.session_state['all_threads'] = retrieve_all_threads()
    if not st.session_state['all_threads']:
        st.session_state['all_threads'] = [str(uuid.uuid4())]
    st.session_state['thread_id'] = st.session_state['all_threads'][0]

if 'store' not in st.session_state:
    st.session_state['store'] = {}

# Sync history from DB for the current thread
current_tid = st.session_state['thread_id']
if current_tid not in st.session_state['store']:
    st.session_state['store'][current_tid] = get_chat_history(current_tid)

# --- Sidebar ---
st.sidebar.title("My Conversations")
if st.sidebar.button("➕ New Chat"):
    new_id = str(uuid.uuid4())
    st.session_state['all_threads'].append(new_id)
    st.session_state['thread_id'] = new_id
    st.rerun()

for tid in st.session_state['all_threads']:
    if st.sidebar.button(f"💬 {tid[:8]}", key=tid):
        st.session_state['thread_id'] = tid
        st.rerun()

# --- Display Messages ---
for message in st.session_state['store'][current_tid]:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# --- User Input ---
user_input = st.chat_input("Type your message...")

if user_input:
    # 1. Update UI locally
    st.session_state['store'][current_tid].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Stream from AI
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        config = {"configurable": {"thread_id": current_tid}}
        
        for chunk, _ in chatbot.stream(
            {'messages': [HumanMessage(content=user_input)]}, 
            config=config, 
            stream_mode="messages"
        ):
            if chunk.content:
                full_response += chunk.content
                placeholder.markdown(full_response + "▌")
        
        placeholder.markdown(full_response)
    
    # 3. Save response locally
    st.session_state['store'][current_tid].append({"role": "assistant", "content": full_response})