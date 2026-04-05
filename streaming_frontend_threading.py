import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage
import uuid

# ************* Utility Functions *************

def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    new_thread = generate_thread_id()
    st.session_state['thread_id'] = new_thread
    st.session_state['store'][new_thread] = []
    st.session_state['all_threads'].append(new_thread)


# ************* Streamlit UI Setup *************

st.set_page_config(page_title="LangGraph Chat", layout="centered")
st.title("🤖 Local AI Chatbot")


# ************* Session State Initialization *************

if 'store' not in st.session_state:
    st.session_state['store'] = {}

if 'thread_id' not in st.session_state:
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    st.session_state['store'][thread_id] = []

if 'all_threads' not in st.session_state:
    st.session_state['all_threads'] = [st.session_state['thread_id']]


# ************* Sidebar (Multi Chat System) *************

st.sidebar.title("LangGraph Chatbot")

# New Chat Button
if st.sidebar.button("➕ New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")

# Show current thread
st.sidebar.text(f"Current: {st.session_state['thread_id'][:8]}")

# Show all threads
for thread in st.session_state['all_threads']:
    if st.sidebar.button(f"💬 {thread[:8]}"):
        st.session_state['thread_id'] = thread


# ************* Load Current Conversation *************

current_thread = st.session_state['thread_id']
messages = st.session_state['store'][current_thread]


# ************* Display Chat Messages *************

for message in messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


# ************* User Input *************

user_input = st.chat_input("Type your message here...")


# ************* Chat Logic (Streaming + Memory) *************

if user_input:
    config = {'configurable': {'thread_id': current_thread}}

    # 1. Store & display user message
    st.session_state['store'][current_thread].append({
        'role': 'user',
        'content': user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Streaming response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        for message_chunk, metadata in chatbot.stream(
            {'messages': [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="messages"
        ):
            if hasattr(message_chunk, "content") and message_chunk.content:
                full_response += message_chunk.content
                response_placeholder.markdown(full_response + "▌")

        # Final response (remove cursor)
        response_placeholder.markdown(full_response)

    # 3. Save assistant response
    st.session_state['store'][current_thread].append({
        'role': 'assistant',
        'content': full_response
    })