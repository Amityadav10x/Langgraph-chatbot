import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="LangGraph Chat", layout="centered")
st.title("🤖 Local AI Chatbot")

CONFIG = {'configurable': {'thread_id': 'thread-1'}}

# Initialize session state
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# Show previous messages
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Input box
user_input = st.chat_input('Type your message here...')

if user_input:
    # 1. Show user message
    st.session_state['message_history'].append({
        'role': 'user',
        'content': user_input
    })
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. STREAMING RESPONSE (UPDATED 🔥)
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        for message_chunk, metadata in chatbot.stream(
            {'messages': [HumanMessage(content=user_input)]},
            config=CONFIG,
            stream_mode="messages"
        ):
            if hasattr(message_chunk, "content") and message_chunk.content:
                full_response += message_chunk.content
                response_placeholder.markdown(full_response + "▌")

        # Final response without cursor
        response_placeholder.markdown(full_response)

    # 3. Save assistant response
    st.session_state['message_history'].append({
        'role': 'assistant',
        'content': full_response
    })

    