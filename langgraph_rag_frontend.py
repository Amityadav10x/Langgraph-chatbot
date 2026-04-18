import streamlit as st
import asyncio
import uuid
from langgraph_rag_backend import chatbot, ingest_pdf, retrieve_all_threads

# --- 1. Page Config & Custom Styling ---
st.set_page_config(page_title="Carrer_OS | Local Agentic RAG", layout="wide")

# Add a bit of professional flair to the header
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #007BFF; }
    .status-text { color: #6c757d; font-style: italic; }
    </style>
    <div class="main-header">🤖 Carrer_OS Local AI</div>
    <p class="status-text">Powered by LangGraph & Ollama (Phi-3)</p>
    """, unsafe_allow_html=True)

# --- 2. State Initialization ---
# This ensures that even on refresh, we don't lose our place
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- 3. Sidebar: Thread & Document Management ---
with st.sidebar:
    st.header("📂 Conversation Hub")
    
    # Dynamic Thread Selector
    existing_threads = retrieve_all_threads()
    all_options = sorted(list(set([st.session_state.thread_id] + existing_threads)))
    
    # We use a callback to clear the UI history when switching threads
    def on_thread_change():
        st.session_state.chat_history = [] 
        # Note: The backend SqliteSaver will handle loading the actual data

    selected_thread = st.selectbox(
        "Current Session ID",
        options=all_options,
        index=all_options.index(st.session_state.thread_id),
        on_change=on_thread_change
    )
    st.session_state.thread_id = selected_thread
    
    st.divider()
    
    st.header("📄 Knowledge Base")
    uploaded_file = st.file_uploader("Upload a PDF for context", type="pdf")
    
    if uploaded_file:
        if st.button("Index Document", use_container_width=True):
            with st.spinner(f"Reading {uploaded_file.name}..."):
                try:
                    # Convert to bytes for our backend 'ingest_pdf' function
                    file_bytes = uploaded_file.read()
                    ingest_pdf(file_bytes, st.session_state.thread_id, uploaded_file.name)
                    st.success("Indexing complete! You can now ask questions.")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

# --- 4. Chat Display Logic ---
# Load existing messages from the graph if UI history is empty (useful for thread switching)
if not st.session_state.chat_history:
    state = chatbot.get_state({"configurable": {"thread_id": st.session_state.thread_id}})
    if state.values and "messages" in state.values:
        for msg in state.values["messages"]:
            # Logic to handle different message types for UI display
            role = "user" if msg.type == "human" else "assistant"
            # We filter out tool-output noise from the UI
            if msg.content and not isinstance(msg.content, list):
                st.session_state.chat_history.append({"role": role, "content": msg.content})

# Render the history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 5. User Interaction & Streaming ---
# --- 5. User Interaction & Streaming ---
if prompt := st.chat_input("Ask about the PDF or perform a calculation..."):
    # Immediate UI update for user message
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process AI Response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = "" 

        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        inputs = {"messages": [("user", prompt)]}
        
        # KEY CHANGE: Use standard .stream() instead of async
        # We don't need 'asyncio.run' or 'async def' anymore!
        for event in chatbot.stream(inputs, config=config, stream_mode="values"):
            if event.get("messages"):
                last_msg = event["messages"][-1]
                # Ensure we only stream text content, not tool metadata
                if last_msg.type == "ai" and last_msg.content:
                    full_response = last_msg.content
                    response_placeholder.markdown(full_response + "▌")
        
        # Final render without the blinking cursor block
        response_placeholder.markdown(full_response)
        
        # Save the AI response to session state history
        if full_response:
            st.session_state.chat_history.append({"role": "assistant", "content": full_response})