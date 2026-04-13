import asyncio
import os
from typing import Annotated, TypedDict
# Change: Import ChatOllama instead of ChatOpenAI
from langchain_ollama import ChatOllama 
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_mcp_adapters.client import MultiServerMCPClient

# 1. Define the State
class State(TypedDict):
    messages: Annotated[list, add_messages]

async def main():
    # 2. Configure MCP Server (Ensure path to math_server.py is correct)
    server_script_path = os.path.abspath("math_server.py")
    
    server_config = {
        "my_math_tools": {
            "command": "python",
            "args": [server_script_path],
            "transport": "stdio"
        }
    }

    # 3. Initialize the MCP Client
    async with MultiServerMCPClient(server_config) as client:
        mcp_tools = await client.get_tools()
        
        # 4. Initialize Ollama LLM
        # Note: Ensure you have run 'ollama pull llama3.1' in your terminal
        llm = ChatOllama(
            model="phi3", 
            temperature=0,
            base_url="http://localhost:11434" # Default Ollama port
        ).bind_tools(mcp_tools)

        # 5. Define Nodes
        async def call_model(state: State):
            response = await llm.ainvoke(state["messages"])
            return {"messages": [response]}

        # 6. Build the Graph
        workflow = StateGraph(State)
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(mcp_tools))

        workflow.add_edge(START, "agent")
        
        def should_continue(state: State):
            last_message = state["messages"][-1]
            if last_message.tool_calls:
                return "tools"
            return END

        workflow.add_conditional_edges("agent", should_continue)
        workflow.add_edge("tools", "agent")

        app = workflow.compile()

        # 7. Execute a Test Query
        query = {"messages": [("user", "My salary is 5000, what is it after 10% tax?")]}
        
        async for event in app.astream(query):
            for value in event.values():
                msg = value["messages"][-1]
                if hasattr(msg, 'content') and msg.content:
                    print(f"\nAssistant: {msg.content}")

if __name__ == "__main__":
    asyncio.run(main())