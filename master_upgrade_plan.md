# 🚀 Multi-Agent Chatbot — Master Upgrade Roadmap
> **From:** Basic LangChain + SQLite chatbot (v1.0)  
> **To:** Production-Ready Agentic RAG System with LLM Evaluation (v4.0)  
> **Methodology:** Incremental versioned releases — test each version before moving to next  
> **Design Level:** High-Level + Low-Level Design (HLD + LLD) — interview-ready

---

## 📋 Table of Contents

1. [Current State Audit & Bug Fixes (v1.0 → v1.1)](#phase-0)
2. [Information You Need to Provide](#info-needed)
3. [System Design: HLD & LLD](#system-design)
4. [Version Roadmap Overview](#version-roadmap)
5. [Phase 1 — Foundation: Structure + Rate Limiter (v1.1 → v2.0)](#phase-1)
6. [Phase 2 — Multi-Agent Core (v2.0 → v2.5)](#phase-2)
7. [Phase 3 — Agentic RAG System (v2.5 → v3.0)](#phase-3)
8. [Phase 4 — Human-in-the-Loop + Evaluation (v3.0 → v4.0)](#phase-4)
9. [Model Selection Guide](#model-selection)
10. [RAG Strategy by Use Case](#rag-strategies)
11. [Free API Sources & Keys Needed](#free-apis)
12. [Folder Structure (Final)](#folder-structure)
13. [Production Checklist](#production-checklist)
14. [Interview Explanation Guide](#interview-guide)

---

## 🔍 Phase 0 — Current State Audit & Bug Fixes {#phase-0}

**Goal:** Fix all bugs in the existing project before touching new features.  
**Checkpoint:** App runs locally, chat works, conversation history persists.

### Bugs Found in `app.py`

#### Bug 1 — `_load_thread` used before definition
```python
# ❌ BROKEN — called in sidebar loop but defined 80 lines later
if st.sidebar.button(label, ...):
    st.session_state.messages = _load_thread(tid)  # NameError!

# ✅ FIX — move _load_thread() to top of file before sidebar code
```

#### Bug 2 — `thread_id` used as bare variable (not from session state)
```python
# ❌ BROKEN in load_conversation()
conversation = chatbot.get_state(config={"configurable":{"thread_id":thread_id}})
# thread_id is not defined in scope here!

# ✅ FIX
conversation = chatbot.get_state(
    config={"configurable": {"thread_id": st.session_state["thread_id"]}}
).values.get("messages", [])
```

#### Bug 3 — `store_info()` crashes if DB is empty on first run
```python
# ❌ BROKEN — checkpointer.list(None) throws on empty DB
def store_info():
    return list({c.config["configurable"]["thread_id"] for c in checkpointer.list(None)})

# ✅ FIX — wrap in try/except
def store_info():
    try:
        return list({c.config["configurable"]["thread_id"] for c in checkpointer.list(None)})
    except Exception:
        return []
```

#### Bug 4 — `stream_mode="messages"` returns `(chunk, metadata)` tuples but content may be empty
```python
# ❌ Can produce empty strings in the stream
ai_message = st.write_stream(
    message_chunk.content for message_chunk, metadata in chatbot.stream(...)
)

# ✅ FIX — filter empty chunks
ai_message = st.write_stream(
    message_chunk.content
    for message_chunk, metadata in chatbot.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config={"configurable": {"thread_id": st.session_state["thread_id"]}},
        stream_mode="messages"
    )
    if message_chunk.content  # skip empty chunks
)
```

#### Bug 5 — `chatbot_agent.py` uses deprecated `ChatOpenAI` from `langchain_community`
```python
# ❌ DEPRECATED — will break with newer langchain
from langchain_community.chat_models import ChatOpenAI

# ✅ FIX
from langchain_openai import ChatOpenAI
```

### Fixed `chatbot_agent.py` (v1.1)
```python
import os
import sqlite3
from typing import Annotated
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI           # ✅ fixed import
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv

load_dotenv()

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

llm = ChatOpenAI(
    model="meta/llama-3.3-70b-instruct:free",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.7,
    streaming=True,
)

def chat_node(state: ChatState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

conn = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
chatbot = graph.compile(checkpointer=checkpointer)

def store_info() -> list[str]:
    try:
        return list({
            c.config["configurable"]["thread_id"]
            for c in checkpointer.list(None)
        })
    except Exception:
        return []
```

### Fixed `app.py` (v1.1)
```python
import uuid
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from chatbot_agent import chatbot, store_info

# ── Helpers (define BEFORE use) ───────────────────────────────────────────────

def _load_thread(thread_id: str) -> list[dict]:
    try:
        state = chatbot.get_state(
            config={"configurable": {"thread_id": thread_id}}
        ).values
        messages = state.get("messages", [])
        result = []
        for m in messages:
            if isinstance(m, HumanMessage):
                result.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                result.append({"role": "assistant", "content": m.content})
        return result
    except Exception:
        return []

def generate_thread_id() -> str:
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    _add_thread(thread_id)
    st.session_state["message_history"] = []

def _add_thread(thread_id: str):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

# ── Session init ──────────────────────────────────────────────────────────────

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = store_info()

_add_thread(st.session_state["thread_id"])

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("Langchain Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()
    st.rerun()

st.sidebar.header("My Conversations")

for tid in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(str(tid)[:18] + "...", key=f"thread_{tid}"):
        st.session_state["thread_id"] = tid
        st.session_state["message_history"] = _load_thread(tid)
        st.rerun()

# ── Chat display ──────────────────────────────────────────────────────────────

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Chat input ────────────────────────────────────────────────────────────────

user_input = st.chat_input("Type here")

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            chunk.content
            for chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config={"configurable": {"thread_id": st.session_state["thread_id"]}},
                stream_mode="messages",
            )
            if chunk.content
        )

    st.session_state["message_history"].append({"role": "assistant", "content": ai_message})
```

### ✅ v1.1 Checkpoint
```bash
pip install langchain langchain-openai langgraph streamlit python-dotenv
streamlit run app.py
# Verify: chat works, history loads, new chat button works
```

---

## ❓ Information You Need to Provide {#info-needed}

Before Phase 2 onwards, gather these free API keys:

| API | What it unlocks | Get it at | Free Tier |
|-----|----------------|-----------|-----------|
| `NVIDIA_API_KEY` | All NVIDIA NIM models (embedding, rerank, LLMs) | [build.nvidia.com](https://build.nvidia.com) | 40 req/min per model |
| `OPENROUTER_API_KEY` | Already have this ✅ | openrouter.ai | Free models available |
| `TAVILY_API_KEY` | Web search for research agent | [tavily.com](https://tavily.com) | 1,000 req/month free |
| `GOOGLE_API_KEY` | Gemini models as fallback + vision | [aistudio.google.com](https://aistudio.google.com) | 15 req/min free |
| `GROQ_API_KEY` | Ultra-fast inference (llama, mistral) | [console.groq.com](https://console.groq.com) | 6,000 req/day free |
| `TOGETHER_API_KEY` *(optional)* | More free model options | [api.together.ai](https://api.together.ai) | $5 free credits |
| `COHERE_API_KEY` *(optional)* | Better reranking model | [dashboard.cohere.com](https://dashboard.cohere.com) | 100 req/min free |

**Action:** Copy `.env.example` below and fill in what you have. You can start with just `NVIDIA_API_KEY` + `OPENROUTER_API_KEY`.

---

## 🏗️ System Design: HLD & LLD {#system-design}

### High-Level Design (HLD)

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI (app.py)                    │
│  [File Upload] [Chat Input] [Feedback Buttons] [Thread Sidebar] │
└────────────────────────────┬────────────────────────────────────┘
                             │ User Message + Optional File
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph Orchestrator                        │
│                      (core/orchestrator.py)                     │
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────────┐  │
│  │ Safety   │──▶│ Router   │──▶│ Agent    │──▶│ Evaluator  │  │
│  │ Node     │   │ Node     │   │ Node     │   │ Node       │  │
│  └──────────┘   └──────────┘   └──────────┘   └────────────┘  │
│                                     │                           │
│                              Human-in-Loop?                     │
│                              (if blocked/stuck)                 │
└─────────────────────────────────────────────────────────────────┘
                             │
        ┌────────────────────┼─────────────────────┐
        ▼                    ▼                      ▼
┌──────────────┐    ┌──────────────┐      ┌──────────────────┐
│  Agent Pool  │    │  RAG Stack   │      │  Tool Layer      │
│              │    │              │      │                  │
│ chat_agent   │    │ Embeddings   │      │ search_tool      │
│ reason_agent │    │ ChromaDB     │      │ file_handler     │
│ coding_agent │    │ Reranker     │      │ code_runner      │
│ research_ag  │    │ Agentic RAG  │      │ web_scraper      │
│ rag_agent    │    │              │      │                  │
│ ocr_agent    │    └──────────────┘      └──────────────────┘
│ safety_agent │
│ eval_agent   │
│ feedback_ag  │
└──────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────┐
│                    Model Layer                        │
│                                                      │
│  NVIDIA NIM ──────── Primary LLMs + Embeddings       │
│  Groq ─────────────── Ultra-fast fallback            │
│  Google AI Studio ─── Vision + Gemini fallback       │
│  OpenRouter ────────── Free model pool               │
└──────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────┐
│                   Memory Layer                        │
│  SQLite (LangGraph checkpointer) ─── Conversation    │
│  ChromaDB ─────────────────────────── RAG vectors    │
└──────────────────────────────────────────────────────┘
```

### Low-Level Design (LLD)

#### AgentState Schema (LangGraph)
```python
class AgentState(TypedDict):
    messages:       Annotated[list[BaseMessage], add_messages]  # Full conversation
    task:           str           # Routing decision: "chat|coding|rag|research|..."
    file_path:      str | None    # Uploaded file name
    file_bytes:     bytes | None  # Raw file bytes for vision/OCR
    blocked:        bool          # Safety check result
    block_reason:   str           # If blocked, why
    hitl_needed:    bool          # Human-in-the-loop trigger
    hitl_question:  str           # What to ask the human
    hitl_response:  str           # Human's answer
    eval_score:     float         # Evaluator score 0.0-1.0
    eval_feedback:  str           # Evaluator's critique
    rag_context:    str           # Retrieved context (for transparency)
    agent_trace:    list[str]     # Which agents ran (for debugging)
```

#### Graph Flow (Complete v4.0)
```
START
  │
  ▼
[safety_node] ─── UNSAFE ──▶ [blocked_node] ──▶ END
  │
  SAFE
  │
  ▼
[router_node]  ← keyword + file-extension routing
  │
  ▼
[agent_node]  ← dispatches to correct agent
  │
  ├── needs web search? → [research_node] → back to agent
  │
  ├── needs clarification? → [hitl_node] → wait for human → back to agent
  │
  ▼
[evaluator_node]  ← scores answer quality
  │
  ├── score < 0.7? → [retry_node] → back to agent (max 2 retries)
  │
  ▼
[feedback_node]  ← formats final response with score metadata
  │
  ▼
END
```

---

## 🗺️ Version Roadmap Overview {#version-roadmap}

| Version | What's Added | Checkpoint Test |
|---------|-------------|-----------------|
| **v1.1** | Bug fixes on existing code | `streamlit run app.py` — chat works |
| **v2.0** | Folder structure + Rate limiter + Logger + NVIDIA models | Import all modules without errors |
| **v2.1** | Safety agent + Task router | Router picks correct agent per input |
| **v2.2** | Chat, Reasoning, Coding agents | 3 agents respond correctly |
| **v2.3** | Research agent (Tavily web search) | "What's the latest news on X?" works |
| **v2.4** | File handler + OCR/Vision agent | Upload PDF/image, ask question |
| **v2.5** | Full orchestrator wiring all agents | All agents accessible from one UI |
| **v3.0** | Naive RAG (vector search) | Upload doc, ask question, get cited answer |
| **v3.1** | Reranker + Advanced RAG | Better precision on doc Q&A |
| **v3.2** | Agentic RAG (self-query, multi-hop) | Multi-doc reasoning works |
| **v3.3** | Hybrid RAG (BM25 + vector) | Better recall for keyword queries |
| **v4.0** | Human-in-the-loop | Bot asks clarifying questions when stuck |
| **v4.1** | LLM Evaluator agent | Every answer gets a quality score |
| **v4.2** | Feedback agent + UI feedback buttons | 👍/👎 captured and logged |
| **v4.3** | Evaluation metrics dashboard | RAGAS scores visible in sidebar |

---

## 📁 Phase 1 — Foundation (v1.1 → v2.0) {#phase-1}

### Task 1.1 — Create Folder Structure
```bash
mkdir -p core agents tools rag memory utils
touch core/__init__.py agents/__init__.py tools/__init__.py
touch rag/__init__.py memory/__init__.py utils/__init__.py
```

### Task 1.2 — `.env.example`
```bash
# ── Primary Models ────────────────────────────────────────────────
OPENROUTER_API_KEY=your-openrouter-key        # Already have this

# ── NVIDIA NIM (40 req/min per model, free) ───────────────────────
NVIDIA_API_KEY=nvapi-your-key-here

# ── Fallback / Alternative Sources ────────────────────────────────
GROQ_API_KEY=your-groq-key                    # 6K req/day free
GOOGLE_API_KEY=your-google-ai-studio-key      # 15 req/min free, Gemini
TOGETHER_API_KEY=your-together-key            # Optional, $5 free credits

# ── Tools ─────────────────────────────────────────────────────────
TAVILY_API_KEY=your-tavily-key                # Web search, 1K/month free

# ── RAG Config ────────────────────────────────────────────────────
CHROMA_PERSIST_DIR=./chroma_db

# ── App Config ────────────────────────────────────────────────────
LOG_LEVEL=INFO
MAX_RETRIES=3
ENABLE_EVAL=true
EVAL_THRESHOLD=0.7
```

### Task 1.3 — `requirements.txt`
```txt
# UI
streamlit>=1.35.0

# LangChain ecosystem
langchain>=0.2.0
langchain-community>=0.2.0
langchain-openai>=0.1.0
langchain-nvidia-ai-endpoints>=0.1.0
langchain-google-genai>=1.0.0
langchain-groq>=0.1.0
langgraph>=0.2.0

# Vector DB
chromadb>=0.5.0

# RAG extras
rank-bm25>=0.2.2                 # BM25 keyword search for hybrid RAG
ragas>=0.1.0                     # RAG evaluation framework

# Tools
tavily-python>=0.3.0

# File parsing
pypdf>=4.0.0
python-docx>=1.1.0
pandas>=2.0.0
pillow>=10.0.0
openpyxl>=3.1.0

# Resilience
tenacity>=8.2.0

# Utils
python-dotenv>=1.0.0
structlog>=24.0.0
```

### Task 1.4 — `utils/logger.py`
```python
import structlog
import logging
import os


def get_logger(name: str):
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level, logging.INFO)
        ),
    )
    return structlog.get_logger(name)
```

### Task 1.5 — `utils/rate_limiter.py`
*(Full sliding-window limiter with fallback chain — from existing MULTIAGENT_UPGRADE.md Step 2)*

Key design decisions to explain in interview:
- **Sliding window** (not fixed window): fairer, no burst at minute boundary
- **Thread-safe** with `threading.Lock()`: Streamlit runs multiple sessions
- **Fallback chain**: degrades gracefully instead of crashing

```python
import time
import threading
from collections import defaultdict, deque
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Model rate limits (NVIDIA NIM free tier: 40 RPM) ─────────────────────────
RATE_LIMITS = {
    "meta/llama-3.3-70b-instruct":            35,
    "deepseek-ai/deepseek-r1":                35,
    "qwen/qwen2.5-coder-32b-instruct":        35,
    "nvidia/llama-3.3-nemotron-super-49b-v1": 35,
    "microsoft/phi-4":                        35,
    "nvidia/nv-embedqa-e5-v5":                35,
    "nvidia/llama-nemotron-rerank-1b-v2":     35,
    "meta/llama-3.2-90b-vision-instruct":     35,
    "meta/llama-3.2-11b-vision-instruct":     35,
    # Groq fallbacks (6K req/day = ~250/hour = ~4/min — conservative)
    "groq/llama-3.3-70b-versatile":           4,
    # Google fallbacks (15 req/min free)
    "gemini-1.5-flash":                       14,
}

# ── Primary → Fallback chain ──────────────────────────────────────────────────
FALLBACK_CHAIN = {
    "meta/llama-3.3-70b-instruct":            "groq/llama-3.3-70b-versatile",
    "deepseek-ai/deepseek-r1":                "nvidia/llama-3.3-nemotron-super-49b-v1",
    "qwen/qwen2.5-coder-32b-instruct":        "microsoft/phi-4",
    "nvidia/llama-3.3-nemotron-super-49b-v1": "meta/llama-3.3-70b-instruct",
    "meta/llama-3.2-90b-vision-instruct":     "meta/llama-3.2-11b-vision-instruct",
    "groq/llama-3.3-70b-versatile":           "gemini-1.5-flash",
    "gemini-1.5-flash":                       "microsoft/phi-4",
}


class RateLimiter:
    """
    Thread-safe sliding-window rate limiter.
    Tracks request timestamps per model over the last 60 seconds.
    """

    def __init__(self):
        self._windows: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def is_available(self, model: str) -> bool:
        limit = RATE_LIMITS.get(model, 35)
        now = time.time()
        with self._lock:
            window = self._windows[model]
            while window and window[0] < now - 60:
                window.popleft()
            return len(window) < limit

    def record_request(self, model: str) -> None:
        with self._lock:
            self._windows[model].append(time.time())

    def get_available_model(self, primary: str) -> str:
        model = primary
        visited: set[str] = set()
        while model and model not in visited:
            if self.is_available(model):
                if model != primary:
                    logger.info("rate_limit_fallback", primary=primary, using=model)
                return model
            visited.add(model)
            model = FALLBACK_CHAIN.get(model, "")
        logger.warning("all_fallbacks_exhausted", primary=primary)
        return "microsoft/phi-4"


rate_limiter = RateLimiter()  # Singleton shared across app
```

### Task 1.6 — `models/nvidia.py`
```python
import os
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, NVIDIARerank
from dotenv import load_dotenv

load_dotenv()

NVIDIA_BASE = "https://integrate.api.nvidia.com/v1"
NVIDIA_KEY  = os.getenv("NVIDIA_API_KEY")
GROQ_KEY    = os.getenv("GROQ_API_KEY")
GOOGLE_KEY  = os.getenv("GOOGLE_API_KEY")


def get_llm(model: str, temperature: float = 0.7) -> ChatOpenAI:
    """Route to correct provider based on model prefix."""
    if model.startswith("groq/"):
        return ChatGroq(
            model=model.replace("groq/", ""),
            groq_api_key=GROQ_KEY,
            temperature=temperature,
        )
    if model.startswith("gemini"):
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=GOOGLE_KEY,
            temperature=temperature,
        )
    # Default: NVIDIA NIM
    return ChatOpenAI(
        model=model,
        openai_api_base=NVIDIA_BASE,
        openai_api_key=NVIDIA_KEY,
        temperature=temperature,
        max_retries=0,
        request_timeout=60,
    )


def get_embeddings() -> NVIDIAEmbeddings:
    return NVIDIAEmbeddings(
        model="nvidia/nv-embedqa-e5-v5",
        api_key=NVIDIA_KEY,
        truncate="END",
    )


def get_reranker() -> NVIDIARerank:
    return NVIDIARerank(
        model="nvidia/llama-nemotron-rerank-1b-v2",
        api_key=NVIDIA_KEY,
        top_n=5,
    )
```

### Task 1.7 — `models/fallback.py`
```python
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError
from models.nvidia import get_llm
from utils.rate_limiter import rate_limiter
from utils.logger import get_logger

logger = get_logger(__name__)


def get_model_with_fallback(primary: str, temperature: float = 0.7):
    model = rate_limiter.get_available_model(primary)
    rate_limiter.record_request(model)
    return get_llm(model, temperature)


@retry(
    retry=retry_if_exception_type(RateLimitError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def invoke_with_retry(llm, messages):
    return llm.invoke(messages)
```

### ✅ v2.0 Checkpoint
```bash
python -c "from utils.logger import get_logger; print('Logger OK')"
python -c "from utils.rate_limiter import rate_limiter; print('Rate limiter OK')"
python -c "from models.nvidia import get_llm; print('Models OK')"
```

---

## 🤖 Phase 2 — Multi-Agent Core (v2.0 → v2.5) {#phase-2}

### Task 2.1 — Safety Agent (v2.1)

**Why first:** Every input must pass safety check before any LLM call.

```python
# agents/safety_agent.py
from langchain_core.messages import HumanMessage
from models.fallback import get_model_with_fallback, invoke_with_retry
from utils.logger import get_logger

logger = get_logger(__name__)

# Fast keyword blocklist — no LLM call needed for obvious violations
BLOCKED_PATTERNS = [
    "how to make a bomb", "how to hack", "child abuse",
    "create malware", "ddos attack", "generate fake id",
]

SAFETY_PROMPT = """You are a content safety classifier. Respond with exactly:
SAFE — if the message is appropriate
UNSAFE: <brief reason> — if it violates guidelines

Message: {message}
Classification:"""


def check_safety(user_input: str) -> tuple[bool, str]:
    """Returns (is_safe, reason). Fast path first, LLM second."""
    lower = user_input.lower()
    for pattern in BLOCKED_PATTERNS:
        if pattern in lower:
            return False, f"Blocked: {pattern}"
    try:
        llm = get_model_with_fallback("meta/llama-3.3-70b-instruct", temperature=0.0)
        prompt = SAFETY_PROMPT.format(message=user_input[:500])
        response = invoke_with_retry(llm, prompt)
        result = response.content.strip()
        if result.startswith("UNSAFE"):
            reason = result.replace("UNSAFE:", "").strip()
            logger.warning("safety_blocked", reason=reason)
            return False, reason
        return True, "safe"
    except Exception as e:
        logger.error("safety_check_error", error=str(e))
        return True, "check skipped"  # Fail open
```

### Task 2.2 — Task Router (v2.1)

```python
# core/router.py
from dataclasses import dataclass

TASK_MODEL_MAP = {
    "chat":      "meta/llama-3.3-70b-instruct",
    "reasoning": "deepseek-ai/deepseek-r1",
    "coding":    "qwen/qwen2.5-coder-32b-instruct",
    "rag":       "nvidia/llama-3.3-nemotron-super-49b-v1",
    "research":  "meta/llama-3.3-70b-instruct",
    "vision":    "meta/llama-3.2-90b-vision-instruct",
    "ocr":       "meta/llama-3.2-90b-vision-instruct",
}

CODE_KEYWORDS = {
    "code", "python", "function", "debug", "error", "fix", "script",
    "class", "import", "sql", "query", "algorithm", "implement",
    "refactor", "write a function", "write a script",
}
REASONING_KEYWORDS = {
    "analyze", "explain why", "compare", "evaluate", "reason",
    "think through", "step by step", "proof", "derive",
    "how does", "why does", "what causes", "research paper",
}
RESEARCH_KEYWORDS = {
    "search", "find", "latest", "current", "news", "today",
    "recent", "what happened", "look up", "browse",
}
VISION_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
DOC_EXTS    = {".pdf", ".docx", ".txt", ".csv", ".xlsx"}


@dataclass
class RoutingDecision:
    task: str
    model: str
    agent: str


def route(user_input: str, file_path: str | None = None) -> RoutingDecision:
    text = user_input.lower()

    if file_path:
        ext = "." + file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
        if ext in VISION_EXTS:
            return RoutingDecision("vision", TASK_MODEL_MAP["vision"], "vision")
        if ext in DOC_EXTS:
            return RoutingDecision("rag", TASK_MODEL_MAP["rag"], "rag")

    if any(k in text for k in CODE_KEYWORDS):
        return RoutingDecision("coding", TASK_MODEL_MAP["coding"], "coding")
    if any(k in text for k in REASONING_KEYWORDS):
        return RoutingDecision("reasoning", TASK_MODEL_MAP["reasoning"], "reasoning")
    if any(k in text for k in RESEARCH_KEYWORDS):
        return RoutingDecision("research", TASK_MODEL_MAP["research"], "research")

    return RoutingDecision("chat", TASK_MODEL_MAP["chat"], "chat")
```

### Task 2.3 — Core Agents (v2.2)

*(Full agent code from MULTIAGENT_UPGRADE.md Step 7 — `chat_agent.py`, `reasoning_agent.py`, `coding_agent.py`, `research_agent.py`, `ocr_agent.py`)*

**Model assignment rationale (for interviews):**

| Agent | Model | Why |
|-------|-------|-----|
| chat | llama-3.3-70b | Best general quality/speed balance |
| reasoning | deepseek-r1 | Chain-of-thought, math, logic |
| coding | qwen2.5-coder-32b | Purpose-built for code |
| rag | nemotron-super-49b | Best at grounded factual answers |
| research | llama-3.3-70b | Good synthesis of web results |
| vision/ocr | llama-3.2-90b-vision | Only free multimodal NVIDIA model |

### Task 2.4 — Orchestrator v2 (v2.5)

```python
# core/orchestrator.py
from typing import Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from core.router import route, RoutingDecision
from agents.safety_agent import check_safety
from agents.chat_agent import run_chat_agent
from agents.reasoning_agent import run_reasoning_agent
from agents.coding_agent import run_coding_agent
from agents.research_agent import run_research_agent
from agents.rag_agent import run_rag_agent
from agents.ocr_agent import run_ocr_agent
from memory.checkpointer import get_checkpointer
from utils.logger import get_logger

logger = get_logger(__name__)


class AgentState(TypedDict):
    messages:       Annotated[list[BaseMessage], add_messages]
    task:           str
    file_path:      str | None
    file_bytes:     bytes | None
    blocked:        bool
    block_reason:   str
    hitl_needed:    bool
    hitl_question:  str
    hitl_response:  str
    eval_score:     float
    eval_feedback:  str
    rag_context:    str
    agent_trace:    list[str]


def safety_node(state: AgentState) -> dict:
    msg = state["messages"][-1].content
    is_safe, reason = check_safety(msg)
    return {"blocked": not is_safe, "block_reason": reason,
            "agent_trace": state.get("agent_trace", []) + ["safety"]}


def router_node(state: AgentState) -> dict:
    msg = state["messages"][-1].content
    decision = route(msg, state.get("file_path"))
    logger.info("routed", task=decision.task, model=decision.model)
    return {"task": decision.task,
            "agent_trace": state.get("agent_trace", []) + [f"router:{decision.task}"]}


def agent_node(state: AgentState) -> dict:
    if state.get("blocked"):
        reply = f"⚠️ I can't help with that. {state.get('block_reason', '')}"
        return {"messages": [AIMessage(content=reply)]}

    task = state.get("task", "chat")
    messages = state["messages"]
    file_bytes = state.get("file_bytes")

    dispatch = {
        "chat":     lambda: run_chat_agent(messages),
        "reasoning":lambda: run_reasoning_agent(messages),
        "coding":   lambda: run_coding_agent(messages),
        "research": lambda: run_research_agent(messages),
        "rag":      lambda: run_rag_agent(messages),
        "vision":   lambda: run_ocr_agent(file_bytes, messages[-1].content) if file_bytes else run_chat_agent(messages),
        "ocr":      lambda: run_ocr_agent(file_bytes, messages[-1].content) if file_bytes else run_chat_agent(messages),
    }

    reply = dispatch.get(task, dispatch["chat"])()
    return {
        "messages": [AIMessage(content=reply)],
        "agent_trace": state.get("agent_trace", []) + [f"agent:{task}"],
    }


def should_block(state: AgentState) -> str:
    return "blocked" if state.get("blocked") else "continue"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("safety", safety_node)
    g.add_node("router", router_node)
    g.add_node("agent",  agent_node)

    g.add_edge(START, "safety")
    g.add_conditional_edges("safety", should_block, {
        "blocked":  "agent",   # agent handles blocked state → returns block message
        "continue": "router",
    })
    g.add_edge("router", "agent")
    g.add_edge("agent", END)

    return g.compile(checkpointer=get_checkpointer())


chatbot = build_graph()
```

### ✅ v2.5 Checkpoint
```bash
streamlit run app.py
# Test: "write a python sort function" → coding agent
# Test: "explain why black holes exist" → reasoning agent
# Test: "how to make explosives" → blocked
# Test: "latest news about AI" → research agent
```

---

## 📚 Phase 3 — Agentic RAG System (v2.5 → v3.0) {#phase-3}

### RAG Strategy by Use Case {#rag-strategies}

| RAG Type | Use Case | When to Use |
|----------|----------|-------------|
| **Naive RAG** | Simple doc Q&A | Single short document, basic questions |
| **Advanced RAG** | Precise doc Q&A with reranking | Multiple docs, specific fact retrieval |
| **Hybrid RAG** | Keyword + semantic search | Technical docs with exact terms (APIs, code) |
| **Multi-hop RAG** | Multi-document reasoning | "Compare X from doc1 with Y from doc2" |
| **Agentic RAG** | Self-directed research | Complex questions requiring iterative search |
| **Graph RAG** | Relationship-heavy data | Org charts, knowledge graphs, ontologies |

### Task 3.1 — Naive RAG (v3.0)

```python
# rag/vector_store.py — from MULTIAGENT_UPGRADE.md Step 8
# (ChromaDB + NVIDIA embeddings + recursive text splitter)
# Checkpoint: upload a PDF, ask a question, get cited answer
```

### Task 3.2 — Advanced RAG with Reranker (v3.1)

```python
# rag/reranker.py — NVIDIA Nemotron reranker
# Pipeline: embed → retrieve top-8 → rerank → keep top-4 → generate
# Checkpoint: precision improves on doc Q&A (fewer hallucinations)
```

### Task 3.3 — Hybrid RAG: BM25 + Vector (v3.2)

```python
# rag/hybrid_retriever.py
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from rag.embeddings import get_embedding_model
from utils.logger import get_logger

logger = get_logger(__name__)


class HybridRetriever:
    """
    Combines BM25 (keyword) and ChromaDB (semantic) retrieval.
    Uses Reciprocal Rank Fusion (RRF) to merge result lists.
    
    Why hybrid?
    - Vector search: good for semantic similarity ("similar concepts")
    - BM25: good for exact keywords ("API endpoint name", "error code 404")
    - Together: best recall for both query types
    """

    def __init__(self, collection_name: str = "default"):
        self.collection_name = collection_name
        self._docs: list[Document] = []
        self._bm25: BM25Okapi | None = None
        self._vector_store: Chroma | None = None

    def add_documents(self, documents: list[Document]):
        self._docs = documents
        tokenized = [doc.page_content.lower().split() for doc in documents]
        self._bm25 = BM25Okapi(tokenized)
        embeddings = get_embedding_model()
        self._vector_store = Chroma.from_documents(
            documents, embeddings, collection_name=self.collection_name
        )
        logger.info("hybrid_index_built", docs=len(documents))

    def retrieve(self, query: str, k: int = 6) -> list[Document]:
        if not self._docs:
            return []

        # BM25 scores
        tokens = query.lower().split()
        bm25_scores = self._bm25.get_scores(tokens)
        bm25_ranked = sorted(
            enumerate(bm25_scores), key=lambda x: x[1], reverse=True
        )[:k * 2]

        # Vector scores
        vector_results = self._vector_store.similarity_search(query, k=k * 2)

        # Reciprocal Rank Fusion (RRF)
        rrf_scores: dict[int, float] = {}
        for rank, (idx, _) in enumerate(bm25_ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (60 + rank + 1)

        for rank, doc in enumerate(vector_results):
            # Find doc index by content match
            for idx, d in enumerate(self._docs):
                if d.page_content == doc.page_content:
                    rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (60 + rank + 1)

        top_indices = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]
        return [self._docs[i] for i in top_indices]
```

### Task 3.4 — Agentic RAG (v3.2)

**Concept:** The agent decides *when* to retrieve, *what* to retrieve, and *whether* the retrieved context is sufficient. If not, it reformulates the query and tries again.

```python
# agents/agentic_rag_agent.py
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from rag.vector_store import similarity_search
from rag.reranker import rerank_documents
from models.fallback import get_model_with_fallback, invoke_with_retry
from utils.logger import get_logger

logger = get_logger(__name__)

MAX_ITERATIONS = 3  # Max retrieval loops before giving final answer

QUERY_REWRITER_PROMPT = """You are a search query optimizer.
Given the user's question and what was retrieved so far, 
write a better search query to find the missing information.

Original question: {question}
Retrieved context so far: {context}
What's still missing: {gap}

Write only the new search query, nothing else:"""

SUFFICIENCY_PROMPT = """You are evaluating whether the context is sufficient to answer the question.

Question: {question}
Context: {context}

Is the context sufficient? Respond with:
SUFFICIENT — if context fully answers the question
INSUFFICIENT: <what is missing> — if more info is needed"""

ANSWER_PROMPT = """Answer the question using only the provided context.
If context is insufficient, say what you know and what you couldn't find.
Always cite the source document when possible.

Context:
{context}

Question: {question}

Answer:"""


def run_agentic_rag(messages: list[BaseMessage], collection_name: str = "default") -> str:
    """
    Iterative retrieval agent:
    1. Retrieve initial docs
    2. Check if sufficient
    3. If not, rewrite query and retrieve again (up to MAX_ITERATIONS)
    4. Generate final answer
    """
    query = messages[-1].content
    llm = get_model_with_fallback("nvidia/llama-3.3-nemotron-super-49b-v1")

    accumulated_context = ""
    queries_tried = [query]

    for iteration in range(MAX_ITERATIONS):
        logger.info("agentic_rag_iteration", iter=iteration + 1, query=query)

        # Step 1: Retrieve
        docs = similarity_search(query, k=6, collection_name=collection_name)
        if not docs:
            if iteration == 0:
                return "No documents in knowledge base. Please upload a document first."
            break

        docs = rerank_documents(query, docs, top_n=4)
        new_context = "\n\n---\n\n".join(d.page_content for d in docs)
        accumulated_context = f"{accumulated_context}\n\n{new_context}".strip()

        # Step 2: Check sufficiency
        check_prompt = SUFFICIENCY_PROMPT.format(
            question=messages[-1].content,
            context=accumulated_context[:3000],
        )
        check_response = invoke_with_retry(llm, check_prompt)
        check_result = check_response.content.strip()

        if check_result.startswith("SUFFICIENT"):
            logger.info("agentic_rag_sufficient", iterations=iteration + 1)
            break

        # Step 3: Extract gap and rewrite query
        gap = check_result.replace("INSUFFICIENT:", "").strip()
        rewrite_prompt = QUERY_REWRITER_PROMPT.format(
            question=messages[-1].content,
            context=accumulated_context[:1000],
            gap=gap,
        )
        new_query_response = invoke_with_retry(llm, rewrite_prompt)
        query = new_query_response.content.strip()
        queries_tried.append(query)
        logger.info("agentic_rag_requeried", new_query=query)

    # Step 4: Generate final answer
    answer_prompt = ANSWER_PROMPT.format(
        context=accumulated_context[:4000],
        question=messages[-1].content,
    )
    final_response = invoke_with_retry(llm, answer_prompt)
    
    result = final_response.content
    if len(queries_tried) > 1:
        result += f"\n\n*Searched {len(queries_tried)} times to find this answer.*"
    
    return result
```

### ✅ v3.0 Checkpoint
```bash
# Test Naive RAG: upload a PDF → ask a specific question from it
# Test Hybrid RAG: ask a query with exact technical terms
# Test Agentic RAG: ask a multi-part question requiring multiple retrieval steps
# Verify: answers cite document sources, no hallucinations outside context
```

---

## 👤 Phase 4 — HITL + Evaluation (v3.0 → v4.0) {#phase-4}

### Task 4.1 — Human-in-the-Loop (v4.0)

**When does HITL trigger?**
- Agent is uncertain (confidence < threshold)
- Task is ambiguous (multiple valid interpretations)
- Agent has been looping (same retrieval > 2 times)
- Safety is borderline (not clearly safe or unsafe)
- User's question is incomplete

```python
# agents/hitl_agent.py
from langchain_core.messages import BaseMessage, AIMessage
from models.fallback import get_model_with_fallback, invoke_with_retry
from utils.logger import get_logger

logger = get_logger(__name__)

CLARIFICATION_PROMPT = """You are a helpful assistant that asks clarifying questions.
The user's request is ambiguous or needs more information.

User's message: {message}
Ambiguity detected: {ambiguity}

Ask a single, clear, specific question to get the information needed.
Keep it concise. Do NOT answer the question — just ask for clarification.

Clarifying question:"""

AMBIGUITY_DETECTOR_PROMPT = """Analyze this user message and determine if it's ambiguous.

User message: {message}

Is this message ambiguous or missing important context?
If YES, respond: AMBIGUOUS: <brief description of what's unclear>
If NO, respond: CLEAR

Your analysis:"""


def detect_ambiguity(user_input: str) -> tuple[bool, str]:
    """Returns (is_ambiguous, description)."""
    try:
        # Use fast model for meta-reasoning
        llm = get_model_with_fallback("microsoft/phi-4", temperature=0.0)
        prompt = AMBIGUITY_DETECTOR_PROMPT.format(message=user_input)
        response = invoke_with_retry(llm, prompt)
        result = response.content.strip()
        if result.startswith("AMBIGUOUS"):
            desc = result.replace("AMBIGUOUS:", "").strip()
            return True, desc
        return False, ""
    except Exception as e:
        logger.error("ambiguity_detection_failed", error=str(e))
        return False, ""


def generate_clarifying_question(user_input: str, ambiguity: str) -> str:
    """Generate a single clarifying question to ask the user."""
    try:
        llm = get_model_with_fallback("meta/llama-3.3-70b-instruct", temperature=0.3)
        prompt = CLARIFICATION_PROMPT.format(
            message=user_input, ambiguity=ambiguity
        )
        response = invoke_with_retry(llm, prompt)
        return response.content.strip()
    except Exception as e:
        logger.error("clarification_failed", error=str(e))
        return "Could you please provide more details about your request?"
```

#### HITL Node in Orchestrator

```python
# Add to core/orchestrator.py

from langgraph.types import interrupt   # LangGraph v0.2+ human-in-the-loop

def hitl_node(state: AgentState) -> dict:
    """
    Check if clarification is needed. If yes, interrupt graph execution
    and surface a question to the user. Resume when user responds.
    """
    from agents.hitl_agent import detect_ambiguity, generate_clarifying_question
    
    last_msg = state["messages"][-1].content
    is_ambiguous, ambiguity_desc = detect_ambiguity(last_msg)
    
    if is_ambiguous and not state.get("hitl_response"):
        # This pauses the graph until the user responds
        question = generate_clarifying_question(last_msg, ambiguity_desc)
        user_response = interrupt(question)   # LangGraph interrupt primitive
        return {
            "hitl_needed": True,
            "hitl_question": question,
            "hitl_response": user_response,
            "agent_trace": state.get("agent_trace", []) + ["hitl"],
        }
    
    return {"hitl_needed": False, "agent_trace": state.get("agent_trace", []) + ["hitl_skip"]}


# Updated graph with HITL node:
# START → safety → router → hitl → agent → evaluator → END
```

#### Streamlit HITL UI Handler

```python
# In app.py — handle HITL interrupts from LangGraph

def run_with_hitl(initial_state: dict, config: dict) -> str:
    """
    Run the graph, handling HITL interrupts gracefully.
    If the graph pauses for human input, show the question in the UI.
    """
    try:
        result = chatbot.invoke(initial_state, config=config)
        
        # Check if the graph was interrupted (waiting for human)
        snapshot = chatbot.get_state(config)
        if snapshot.next:  # Graph is paused mid-execution
            hitl_question = snapshot.values.get("hitl_question", "")
            if hitl_question:
                st.session_state["hitl_pending"] = True
                st.session_state["hitl_question"] = hitl_question
                return f"🤔 **I need clarification:** {hitl_question}"
        
        ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
        return ai_messages[-1].content if ai_messages else "No response."
        
    except Exception as e:
        logger.error("hitl_invoke_failed", error=str(e))
        return f"Error: {str(e)}"

# In the chat input handler:
if st.session_state.get("hitl_pending") and user_input:
    # Resume the paused graph with the human's answer
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    chatbot.update_state(config, {"hitl_response": user_input})
    result = chatbot.invoke(None, config=config)   # Resume from checkpoint
    st.session_state["hitl_pending"] = False
```

### Task 4.2 — LLM Evaluator Agent (v4.1)

**Metrics Implemented:**
- **Answer Relevance** — Is the answer relevant to the question?
- **Faithfulness** — Does the answer stick to the retrieved context (no hallucinations)?
- **Completeness** — Does it fully address all parts of the question?
- **Conciseness** — Is it appropriately concise?

```python
# agents/evaluator_agent.py
import json
from langchain_core.messages import BaseMessage
from models.fallback import get_model_with_fallback, invoke_with_retry
from utils.logger import get_logger

logger = get_logger(__name__)

EVALUATOR_PROMPT = """You are an expert QA evaluator for AI systems.
Evaluate the assistant's response on these dimensions.
Return ONLY valid JSON, no extra text.

Question: {question}
Context provided to AI: {context}
AI's answer: {answer}

Evaluate and return JSON:
{{
  "answer_relevance": <0.0-1.0>,
  "faithfulness": <0.0-1.0>,
  "completeness": <0.0-1.0>,
  "conciseness": <0.0-1.0>,
  "overall_score": <0.0-1.0>,
  "critique": "<1-2 sentence assessment>",
  "needs_retry": <true|false>
}}

Scoring guide:
- answer_relevance: Does the answer directly address the question?
- faithfulness: Does the answer only use facts from the context (no hallucinations)?
- completeness: Does it address all parts of a multi-part question?
- conciseness: Is the answer appropriately brief, not padded?
- overall_score: Weighted average (faithfulness has 40% weight)
- needs_retry: true if overall_score < 0.65"""


from dataclasses import dataclass

@dataclass
class EvalResult:
    answer_relevance: float = 0.0
    faithfulness: float = 0.0
    completeness: float = 0.0
    conciseness: float = 0.0
    overall_score: float = 0.0
    critique: str = ""
    needs_retry: bool = False


def evaluate_response(
    question: str,
    answer: str,
    context: str = "",
) -> EvalResult:
    """Score an AI response across 4 quality dimensions."""
    try:
        # Use a different model than the one that generated the answer
        # to avoid self-serving bias
        llm = get_model_with_fallback("microsoft/phi-4", temperature=0.0)
        prompt = EVALUATOR_PROMPT.format(
            question=question,
            context=context[:2000] if context else "No retrieved context.",
            answer=answer[:2000],
        )
        response = invoke_with_retry(llm, prompt)
        raw = response.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()

        data = json.loads(raw)
        return EvalResult(**data)

    except Exception as e:
        logger.error("evaluator_failed", error=str(e))
        return EvalResult(overall_score=1.0, critique="Evaluation unavailable")


def evaluator_node(state: dict) -> dict:
    """LangGraph node — evaluates the last AI message."""
    messages = state.get("messages", [])
    ai_messages = [m for m in messages if isinstance(m, AIMessage)]
    human_messages = [m for m in messages if isinstance(m, HumanMessage)]

    if not ai_messages or not human_messages:
        return {"eval_score": 1.0, "eval_feedback": ""}

    question = human_messages[-1].content
    answer = ai_messages[-1].content
    context = state.get("rag_context", "")

    result = evaluate_response(question, answer, context)

    logger.info(
        "eval_complete",
        score=result.overall_score,
        faithfulness=result.faithfulness,
        needs_retry=result.needs_retry,
    )

    return {
        "eval_score": result.overall_score,
        "eval_feedback": result.critique,
        "agent_trace": state.get("agent_trace", []) + [f"eval:{result.overall_score:.2f}"],
    }
```

### Task 4.3 — Feedback Agent + UI Buttons (v4.2)

```python
# agents/feedback_agent.py
import json
import sqlite3
from datetime import datetime
from utils.logger import get_logger

logger = get_logger(__name__)

_conn: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect("feedback.db", check_same_thread=False)
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id   TEXT NOT NULL,
                question    TEXT NOT NULL,
                answer      TEXT NOT NULL,
                thumbs_up   INTEGER DEFAULT 0,
                thumbs_down INTEGER DEFAULT 0,
                user_comment TEXT,
                eval_score  REAL,
                timestamp   TEXT NOT NULL
            )
        """)
        _conn.commit()
    return _conn


def log_feedback(
    thread_id: str,
    question: str,
    answer: str,
    thumbs_up: bool,
    comment: str = "",
    eval_score: float = 0.0,
):
    """Log user feedback to SQLite. Called from Streamlit UI on 👍/👎."""
    try:
        conn = _get_conn()
        conn.execute(
            """INSERT INTO feedback 
               (thread_id, question, answer, thumbs_up, thumbs_down, user_comment, eval_score, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                thread_id, question[:500], answer[:1000],
                1 if thumbs_up else 0,
                0 if thumbs_up else 1,
                comment, eval_score,
                datetime.now().isoformat(),
            ),
        )
        conn.commit()
        logger.info("feedback_logged", thumbs_up=thumbs_up, score=eval_score)
    except Exception as e:
        logger.error("feedback_log_failed", error=str(e))


def get_feedback_stats() -> dict:
    """Return aggregate feedback statistics for the dashboard."""
    try:
        conn = _get_conn()
        cursor = conn.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(thumbs_up) as positive,
                SUM(thumbs_down) as negative,
                AVG(eval_score) as avg_eval_score
            FROM feedback
        """)
        row = cursor.fetchone()
        return {
            "total": row[0] or 0,
            "positive": row[1] or 0,
            "negative": row[2] or 0,
            "avg_eval_score": round(row[3] or 0.0, 2),
            "satisfaction_rate": round((row[1] or 0) / max(row[0], 1) * 100, 1),
        }
    except Exception:
        return {"total": 0, "positive": 0, "negative": 0, "avg_eval_score": 0.0}
```

#### Streamlit Feedback UI

```python
# In app.py — add after each assistant message

def render_feedback_buttons(question: str, answer: str, eval_score: float):
    """Render 👍/👎 buttons and optional comment box after each answer."""
    col1, col2, col3 = st.columns([1, 1, 8])
    
    feedback_key = f"feedback_{hash(answer)}"
    
    if feedback_key not in st.session_state:
        with col1:
            if st.button("👍", key=f"up_{feedback_key}"):
                log_feedback(
                    st.session_state.thread_id, question, answer,
                    thumbs_up=True, eval_score=eval_score
                )
                st.session_state[feedback_key] = "positive"
                st.success("Thanks!")
        with col2:
            if st.button("👎", key=f"down_{feedback_key}"):
                st.session_state[feedback_key] = "negative_pending"
        
        if st.session_state.get(feedback_key) == "negative_pending":
            comment = st.text_input("What went wrong?", key=f"comment_{feedback_key}")
            if st.button("Submit", key=f"submit_{feedback_key}"):
                log_feedback(
                    st.session_state.thread_id, question, answer,
                    thumbs_up=False, comment=comment, eval_score=eval_score
                )
                st.session_state[feedback_key] = "negative"
                st.info("Feedback recorded. Thank you!")
    else:
        with col1:
            st.write("✅ Feedback recorded")
```

### Task 4.4 — RAGAS Evaluation Dashboard (v4.3)

```python
# utils/eval_dashboard.py
import streamlit as st
from agents.feedback_agent import get_feedback_stats


def render_eval_sidebar():
    """Show evaluation metrics in Streamlit sidebar."""
    stats = get_feedback_stats()
    
    with st.sidebar:
        st.divider()
        st.subheader("📊 Quality Metrics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Satisfaction", f"{stats['satisfaction_rate']}%")
        with col2:
            st.metric("Avg Eval Score", f"{stats['avg_eval_score']:.2f}")
        
        st.caption(
            f"Based on {stats['total']} responses "
            f"({stats['positive']} 👍 / {stats['negative']} 👎)"
        )
```

### ✅ v4.0 Checkpoint
```bash
# Test HITL: send "tell me about it" (ambiguous) → bot should ask "about what?"
# Test Evaluator: all responses should show an eval score (0.0-1.0) in logs
# Test Feedback: click 👍/👎, verify logged in feedback.db
# Test Dashboard: sidebar shows satisfaction rate and avg eval score
```

---

## 🧠 Model Selection Guide {#model-selection}

### By Task (Free Tier — 2025)

| Task | Primary Model | Source | Latency | Why |
|------|-------------|--------|---------|-----|
| General Chat | `meta/llama-3.3-70b-instruct` | NVIDIA NIM | ~3s | Best quality/speed |
| Deep Reasoning | `deepseek-ai/deepseek-r1` | NVIDIA NIM | ~20s | Chain-of-thought |
| Code Generation | `qwen/qwen2.5-coder-32b-instruct` | NVIDIA NIM | ~5s | Purpose-built |
| RAG / Grounded QA | `nvidia/llama-3.3-nemotron-super-49b-v1` | NVIDIA NIM | ~6s | Best factual accuracy |
| Vision / OCR | `meta/llama-3.2-90b-vision-instruct` | NVIDIA NIM | ~8s | Only free multimodal |
| Meta-reasoning / Eval | `microsoft/phi-4` | NVIDIA NIM | ~2s | Fast, smart small model |
| Fallback (speed) | `llama-3.3-70b-versatile` | Groq | ~0.5s | Fastest available |
| Fallback (vision) | `gemini-1.5-flash` | Google AI Studio | ~2s | Multimodal backup |
| Embeddings | `nvidia/nv-embedqa-e5-v5` | NVIDIA NIM | ~0.5s | Best NVIDIA embedding |
| Reranking | `nvidia/llama-nemotron-rerank-1b-v2` | NVIDIA NIM | ~0.5s | Free reranker |

### Latency Budget Per Task
```
Fast responses (< 3s):    chat, safety check, routing, evaluation
Medium (3-8s):            coding, rag, research
Slow (15-25s):            deepseek-r1 reasoning (warn user with spinner)
```

---

## 🔌 Free API Sources & Additional Recommendations {#free-apis}

### APIs You Should Add

| API | Why | Priority |
|-----|-----|----------|
| **NVIDIA NIM** | Primary LLMs + embeddings + reranker — best free tier | 🔴 Required |
| **Groq** | Fastest fallback, 6K req/day free | 🔴 Required |
| **Tavily** | Best web search for research agent, structured results | 🔴 Required |
| **Google AI Studio** | Gemini fallback + vision backup | 🟡 High |
| **Cohere** | Better reranker than NVIDIA's free one | 🟡 High |
| **Together AI** | More free model options | 🟢 Optional |
| **Jina AI** | Free web reader API (better than raw scraping) | 🟢 Optional |
| **Serper.dev** | Google search API, 2500 free/month | 🟢 Optional |

### Why Each Source Matters

**NVIDIA NIM:** The entire architecture centers on this — embeddings, reranker, and 5+ LLMs all from one key. The `nv-embedqa-e5-v5` embedding model is specifically tuned for retrieval tasks.

**Groq:** When NVIDIA hits rate limits, Groq runs llama-3.3-70b at ~500 tokens/second — orders of magnitude faster than any other free provider. Critical for production.

**Cohere Rerank:** `rerank-english-v3.0` consistently outperforms NVIDIA's 1B reranker on most benchmarks. Free tier: 100 req/min. Worth adding for better RAG precision.

**Jina AI Reader:** `r.jina.ai/{url}` converts any webpage to clean markdown. No API key needed. Use in research agent instead of raw HTML scraping.

```python
# In research_agent.py — add Jina Reader for cleaner web content
import httpx

def fetch_url_content(url: str) -> str:
    """Use Jina Reader for clean webpage content (no API key needed)."""
    try:
        response = httpx.get(f"https://r.jina.ai/{url}", timeout=10)
        return response.text[:3000]
    except Exception:
        return ""
```

---

## 📂 Final Folder Structure {#folder-structure}

```
langchain-chatbot/
├── app.py                          # Streamlit UI (main entry point)
├── .env                            # API keys (gitignored)
├── .env.example                    # Commit this
├── .gitignore
├── requirements.txt
├── README.md
│
├── core/
│   ├── __init__.py
│   ├── router.py                   # Keyword-based task router
│   └── orchestrator.py             # LangGraph graph builder + chatbot object
│
├── models/
│   ├── __init__.py
│   ├── nvidia.py                   # LLM clients (NVIDIA, Groq, Google)
│   └── fallback.py                 # Rate-aware model selection + retry
│
├── agents/
│   ├── __init__.py
│   ├── safety_agent.py             # Content safety (runs first)
│   ├── chat_agent.py               # General chat
│   ├── reasoning_agent.py          # DeepSeek R1 chain-of-thought
│   ├── coding_agent.py             # Qwen Coder
│   ├── research_agent.py           # Tavily web search + synthesis
│   ├── rag_agent.py                # Standard RAG (vector + rerank)
│   ├── agentic_rag_agent.py        # Self-querying iterative RAG
│   ├── ocr_agent.py                # Vision / OCR (Llama Vision)
│   ├── hitl_agent.py               # Human-in-the-loop clarification
│   ├── evaluator_agent.py          # LLM-based response scorer
│   └── feedback_agent.py           # User feedback logger
│
├── rag/
│   ├── __init__.py
│   ├── embeddings.py               # NVIDIA embedding model singleton
│   ├── vector_store.py             # ChromaDB operations
│   ├── reranker.py                 # NVIDIA reranker
│   └── hybrid_retriever.py         # BM25 + vector fusion
│
├── tools/
│   ├── __init__.py
│   ├── search_tool.py              # Tavily wrapper
│   └── file_handler.py             # PDF/DOCX/CSV/TXT extraction
│
├── memory/
│   ├── __init__.py
│   └── checkpointer.py             # SQLite LangGraph checkpointer singleton
│
└── utils/
    ├── __init__.py
    ├── rate_limiter.py             # Sliding window per-model limiter
    ├── logger.py                   # Structlog setup
    └── eval_dashboard.py           # Streamlit metrics widget
```

---

## ✅ Production Checklist {#production-checklist}

### Security
- [ ] `.env` in `.gitignore` — never committed
- [ ] `.env.example` committed with placeholder values
- [ ] No API keys hardcoded anywhere in Python files
- [ ] `feedback.db` in `.gitignore`

### Reliability
- [ ] Rate limiter active (`utils/rate_limiter.py` singleton)
- [ ] Fallback chain covers every primary model
- [ ] All agents wrapped in `try/except` — return strings, never raise
- [ ] `invoke_with_retry` used for all LLM calls (exponential backoff)
- [ ] HITL node prevents infinite loops (max iterations set)

### Quality
- [ ] Safety agent runs before every user message
- [ ] Evaluator agent scores every response
- [ ] Feedback buttons on every AI response
- [ ] RAG answers cite source document
- [ ] Agentic RAG: max 3 retrieval iterations before fallback answer

### Performance
- [ ] Embedding model singleton (loaded once, not per request)
- [ ] Reranker singleton
- [ ] ChromaDB persisted to disk (not in-memory)
- [ ] SQLite `check_same_thread=False` for concurrent sessions

### Observability
- [ ] Structlog in every module
- [ ] `agent_trace` list in state (shows which agents ran)
- [ ] Eval scores logged per response
- [ ] Feedback stats visible in sidebar dashboard

### Git Workflow
```bash
git tag -a v1.0 -m "v1.0: original"
git checkout -b develop
git checkout -b feat/v1.1-bugfixes    # → merge to develop → test
git checkout -b feat/v2.0-foundation  # → merge to develop → test
git checkout -b feat/v2.5-agents      # → merge to develop → test
git checkout -b feat/v3.0-rag         # → merge to develop → test
git checkout -b feat/v4.0-hitl-eval   # → merge to develop → test
git checkout main && git merge develop
git tag -a v4.0 -m "v4.0: production"
```

---

## 🎤 Interview Explanation Guide {#interview-guide}

### "Walk me through your system architecture"

> "The system is a multi-agent chatbot built on LangGraph for orchestration. Every user message flows through a 5-node graph: safety check, task routing, optional HITL clarification, the appropriate agent, and finally an LLM evaluator that scores the response quality. The agents are specialized — there's a coding agent using Qwen-Coder, a reasoning agent on DeepSeek R1, and a RAG agent that uses NVIDIA's embedding and reranking models with ChromaDB as the vector store."

### "How does your RAG system work?"

> "I implemented three RAG strategies. Naive RAG for simple queries: embed the document, do vector similarity search, retrieve top-k, generate. Advanced RAG adds a reranker — I use NVIDIA's Nemotron 1B reranker to reorder retrieved chunks by actual relevance before passing to the LLM. For complex questions I use Agentic RAG: the agent retrieves, checks if the context is sufficient using a separate LLM call, and if not, it reformulates the query and retrieves again — up to 3 iterations. I also have Hybrid RAG combining BM25 keyword search with vector search using Reciprocal Rank Fusion."

### "How do you handle rate limits?"

> "I built a thread-safe sliding window rate limiter. It tracks request timestamps per model in a deque over the last 60 seconds. Before every LLM call, we check if the model is available; if not, we walk a fallback chain — NVIDIA → Groq → Google. The limiter is a singleton shared across Streamlit sessions. On top of that, all LLM calls use exponential backoff retry via Tenacity for transient 429 errors."

### "What's Human-in-the-Loop?"

> "I use LangGraph's `interrupt` primitive. The HITL node detects ambiguous queries using a fast classifier model, generates a clarifying question, and pauses graph execution. Streamlit detects the paused state via `chatbot.get_state()`, displays the question, and when the user answers, calls `chatbot.update_state()` to inject the answer, then resumes from the checkpoint. This prevents the agent from making assumptions and looping on bad data."

### "How do you measure answer quality?"

> "I have a dedicated evaluator agent that runs after every response. It uses a different model than the generator (to avoid self-serving bias) and scores on four dimensions: answer relevance, faithfulness to retrieved context, completeness, and conciseness. Faithfulness has the highest weight at 40% because hallucination is the most critical failure mode for RAG systems. If the overall score is below 0.65, the orchestrator retries generation with a more detailed prompt. All scores are logged and aggregated in a sidebar dashboard."

---

## 📝 Notes & Warnings

### Known Limitations to Address
1. **DeepSeek R1 latency (~20s):** Show spinner with explicit warning: *"Deep reasoning mode — this may take 15-25 seconds"*
2. **ChromaDB thread safety:** Use one collection per `thread_id`, not a global collection
3. **SQLite growth:** Add periodic cleanup: `DELETE FROM checkpoints WHERE thread_ts < <90_days_ago>`
4. **Streamlit re-run:** Clear `file_bytes` from session state after first use
5. **NVIDIA free tier:** 40 RPM limit is per model but shared across your API key — monitor actual usage

### What This Project Demonstrates (for CV/Resume)
- LangGraph state machine design
- Multi-provider LLM orchestration with graceful degradation
- Four RAG architectures (Naive, Advanced, Hybrid, Agentic)
- Human-in-the-loop using LangGraph's interrupt primitive
- LLM-based evaluation pipeline (faithfulness, relevance, completeness)
- Thread-safe rate limiting with sliding window algorithm
- Structured logging with Structlog
- Singleton patterns for expensive ML model initialization
- Incremental versioned development with feature branches
