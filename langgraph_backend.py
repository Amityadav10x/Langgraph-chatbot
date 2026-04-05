from langgraph.graph import StateGraph, START, END, add_messages
from typing import TypedDict, Annotated
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage , HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

# Initialize the model
llm = ChatOllama(model="phi3")

# Define the State
class ChatState(TypedDict):
    # Annotated with add_messages tells LangGraph to append 
    # new messages to the history automatically.
    messages: Annotated[list[BaseMessage], add_messages]

# Define the logic node
def chat_node(state: ChatState):
    # Send the existing message list to Ollama
    response = llm.invoke(state['messages'])
    # IMPORTANT: Return only the NEW message. 
    # LangGraph's add_messages will merge it into the state for you.
    return {"messages": [response]}

# Initialize Memory
checkpoint = InMemorySaver()

# Build the Graph
workflow = StateGraph(ChatState)
workflow.add_node("chat", chat_node)
workflow.add_edge(START, "chat")
workflow.add_edge("chat", END)

# Compile with the correct argument name: 'checkpointer'
chatbot = workflow.compile(checkpointer=checkpoint)


for message_chunk, metadata in chatbot.stream(
    {'messages': [HumanMessage(content="Hello, how are you?")]},
    config={'configurable': {'thread_id': 'thread-1'}},
    stream_mode='messages'
):
    if message_chunk.content:
        print(message_chunk.content, end=" ", flush=True)

print(type(message_chunk))

