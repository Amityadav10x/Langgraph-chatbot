Multi-Agentic AI Orchestrator with LangGraph
A modular, scalable AI Agent architecture built using LangGraph. This project demonstrates the evolution from a simple chatbot to a sophisticated agentic system capable of autonomous tool use, grounded knowledge retrieval (RAG), and human-governed safety protocols.

🚀 Key Features
Agentic Reasoning Loop: Implements the ReAct (Reasoning + Acting) pattern using ToolNode and ToolsCondition to allow the LLM to autonomously decide when to execute external actions.

Modular Subgraphs: Architecture broken down into independent, reusable subgraphs to ensure Failure Isolation and State Separation. Features a dedicated Translation Subgraph for multi-language handoffs.

Model Context Protocol (MCP): A decoupled tool integration layer that uses a standardized "handshake" to connect with local and remote servers, reducing technical debt and integration fatigue.

Agentic RAG (Retrieval-Augmented Generation): High-precision document retrieval using FAISS vector storage. The agent treats the retriever as a tool, deciding when to search private data versus general knowledge.

Human-in-the-Loop (HITL): Safety-first design featuring State-Persistent Interrupts. High-stakes actions are paused for human approval or correction before execution.

Async Execution Layer: Fully refactored for non-blocking operations using async/await and aiosqlite for persistent state management.

🏗️ Architecture
The system is built on a Cyclic Graph architecture, moving away from linear chains to allow the agent to self-correct and iterate on tool outputs.

Input Node: Captures user intent and hydrates the global state.

Reasoning Node (Ollama): Processes the state and generates either a final response or a tool call.

Router (ToolsCondition): Determines the next path based on the LLM's output.

Execution Engine (ToolNode / Subgraphs): Performs the action (Search, Math, RAG, or Subgraph workflows) and returns results to the LLM for synthesis.

🛠️ Tech Stack
Orchestration: LangGraph, LangChain

LLM Engine: Ollama (Phi-3 / Qwen)

Vector Database: FAISS

Database (Persistence): aiosqlite

Frontend: Streamlit

Protocols: MCP (Model Context Protocol)

📦 Setup & Installation
Clone the repository:

Bash
git clone https://github.com/your-username/langgraph-agentic-ai.git
cd langgraph-agentic-ai
Install dependencies:

Bash
pip install -r requirements.txt


3.  **Local LLM Setup:**
    Ensure **Ollama** is running locally with the required models:
    ```bash
    ollama run phi3
    ```

4.  **Run the Application:**
    ```bash
    streamlit run app.py
    ```

## 📂 Project Structure
```text
├── agents/
│   ├── parent_graph.py      # Core orchestration logic
│   └── subgraphs/           # Modular agents (Translator, Searcher, etc.)
├── tools/
│   ├── mcp_client.py        # Model Context Protocol integration
│   └── rag_tool.py          # FAISS retriever logic
├── state/
│   └── checkpoint_db.py     # Async persistence with aiosqlite
└── app.py                   # Streamlit UI
💡 Engineering Insights
State Hydration: Every node in the graph contributes to a unified state, allowing for complex multi-turn reasoning.

Decoupled Tools: By moving tool logic to MCP servers, the core agent remains lightweight and resilient to API changes.

Observability: The modular design allows for granular tracing of each subgraph execution, making debugging significantly easier as the system grows.


Amit Yadav

Data & AI Analyst | AI Engineering Learning Journey
