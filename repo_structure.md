# Cognibot Repository Structure & Application Overview

Cognibot is a powerful Multi-Agent conversational system built using LangGraph, FastAPI, and Streamlit. It intelligently routes user queries to specialized LLM agents (Reasoning, Coding, Research, Vision) and maintains a persistent long-term memory.

---

## 🏗️ Folder Structure

```text
Cognibot/
│
├── backend/                  # FastAPI Application Entrypoint
│   └── main.py               # Exposes the chat streaming API, thread management, and db initialization
│
├── frontend/                 # Streamlit UI
│   └── app.py                # Connects to backend/main.py; handles chat UI, file uploads, and live streaming
│
├── core/                     # Core Orchestration & Database Logic
│   ├── orchestrator.py       # Defines the LangGraph StateGraph (routes user inputs between specialized agents)
│   ├── db.py                 # PostgreSQL connection pool and query helpers for users, threads, and memories
│   └── config.py             # (If present) Centralized configuration variables
│
├── agents/                   # Specialized AI Agents
│   ├── router.py             # Analyzes user input and determines which specialized agent to activate
│   ├── chat_agent.py         # Handles general conversation and injects long-term memory facts into the prompt
│   ├── reasoning_agent.py    # Handles complex logic, math, and multi-step deduction
│   ├── coding_agent.py       # Writes and executes Python code
│   ├── research_agent.py     # Performs web searches to retrieve up-to-date information
│   ├── vision_agent.py       # Processes uploaded images and answers questions about them
│   ├── safety_agent.py       # Validates inputs/outputs to prevent prompt injection and harmful content
│   ├── memory_agent.py       # Asynchronous agent that reads chat history to extract and save long-term facts about the user
│   └── tool_node.py          # Wrapper for executing tool/function calls within the LangGraph architecture
│
├── tools/                    # Tool Implementations for Agents
│   ├── python_repl.py        # Safe execution environment for the coding agent
│   ├── web_search.py         # Integration with Tavily/DuckDuckGo for the research agent
│   └── github_tool.py        # Allows agents to interact with GitHub
│
├── models/                   # LLM Provider Configurations
│   ├── anthropic.py          # Configuration for Claude models
│   ├── nvidia.py             # Configuration for NVIDIA / Llama models
│   └── fallback.py           # Fallback logic if the primary LLM provider fails
│
├── utils/                    # Shared Utilities
│   └── rate_limiter.py       # Redis-based rate limiting to protect the backend endpoints
│
├── docker-compose.yml        # Orchestrates the Postgres, Redis, Backend, and Frontend containers
├── Dockerfile.backend        # Docker build instructions for the FastAPI server
├── Dockerfile.frontend       # Docker build instructions for the Streamlit UI
├── requirements.txt          # Python dependencies
└── .env                      # Environment variables (API Keys, Database URLs)
```

---

## ⚙️ How the Application Works

### 1. The Client (Frontend)
Users interact with the system via the Streamlit web interface (`frontend/app.py`). When a user types a message or uploads an image, the frontend makes a streaming POST request to the FastAPI backend (`backend/main.py`). 

### 2. The Entrypoint (Backend)
FastAPI receives the request, associates it with a specific `thread_id` (for chat history tracking), and passes it to the **Orchestrator**.

### 3. The Orchestrator (LangGraph)
The core of the application resides in `core/orchestrator.py`. It uses LangGraph to define a finite state machine:
- **State Initialization:** The orchestrator fetches the user's past messages from the database.
- **Safety Check:** The input is first routed to the `safety_agent` to ensure it isn't malicious.
- **Routing:** The `router` analyzes the intent and sends the message to the appropriate specialized agent (Chat, Coding, Reasoning, Research, or Vision).
- **Execution:** The specialized agent processes the request. If it needs to use a tool (like searching the web or running code), it routes to the `tool_node`, gets the result, and continues thinking.
- **Memory Extraction:** Once the final response is generated, the graph routes to the `memory_agent` in the background, which silently extracts new facts about the user and saves them to the PostgreSQL database.

### 4. The Output
The final text stream is yielded back to the FastAPI backend, which instantly streams the tokens to the Streamlit UI using Server-Sent Events (SSE). 

### 5. Infrastructure
- **PostgreSQL:** Stores threads (chat history) and user_memory (facts extracted by the memory agent).
- **Redis:** Used by the rate limiter to prevent API abuse.
- **Docker:** Wraps all services into an easily deployable, isolated environment.
