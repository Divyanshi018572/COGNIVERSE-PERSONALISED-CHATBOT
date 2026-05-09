# High Level System Design (Google-Grade)

### Section 1 — System Classification
- **Type:** Real-time conversational inference with stateful orchestrator.
- **Online learning:** No.
- **Use-case category:** Agentic Conversational AI / LLM Orchestration / Generative AI.

### Section 2 — Architecture Diagram (MANDATORY)

```mermaid
graph TD
    User["👤 User / Client"]
    UI["🖥 Streamlit Frontend"]
    API["⚙️ FastAPI Gateway"]
    Orchestrator["🧠 LangGraph Orchestrator"]
    Router["🔀 Intent Router"]
    
    subgraph Agents
        Safety["🛡️ Safety Agent"]
        Chat["💬 Chat Agent"]
        Reasoning["🧠 Reasoning Agent (DeepSeek)"]
        Coding["💻 Coding Agent (Qwen)"]
        Research["🔍 Research Agent (Tavily)"]
    end
    
    LLM["🤖 External Model APIs (NVIDIA NIM/Groq/OpenRouter)"]
    Memory["🗄 State DB (SQLite Volumed)"]
    Search["🌐 Web Search API (Tavily)"]

    User -->|Sends Message| UI
    UI -->|HTTP POST (JSON)| API
    API -->|Invoke State| Orchestrator
    Orchestrator -->|Read/Write Thread| Memory
    Orchestrator --> Safety
    Safety --> Router
    Router --> Chat
    Router --> Reasoning
    Router --> Coding
    Router --> Research
    
    Chat & Reasoning & Coding & Research -->|Prompt| LLM
    Research -->|Search Query| Search
    
    LLM -->|Stream Tokens| Orchestrator
    Orchestrator -->|SSE Tokens| API
    API -->|SSE Tokens| UI
```
*Note: Latency expectation for routing/DB retrieval is <50ms. TTFT from LLM is expected <500ms.*

### Section 3 — Component Design
1. **Frontend Layer (Streamlit):**
   - *Responsibility:* Render UI and maintain local `thread_id` session state.
   - *Tech Choice:* Streamlit (fast prototyping).
   - *Failure Mode:* Connection loss to backend. *Mitigation:* Error boundary and visual feedback.
2. **API Gateway (FastAPI):**
   - *Responsibility:* Expose non-blocking, asynchronous endpoints. Stream LLM tokens to clients.
   - *Tech Choice:* FastAPI + Uvicorn.
   - *Failure Mode:* Thread starvation. *Mitigation:* Use native `async def` and asynchronous generators.
3. **Orchestrator (LangGraph):**
   - *Responsibility:* Manage conversational state machine and agent transitions.
   - *Tech Choice:* LangGraph.
   - *Failure Mode:* Unbounded context window.
4. **Agent Pool:**
   - *Responsibility:* Execute specialized logic based on intent.
   - *Tech Choice:* Specialized models (DeepSeek-R1 for reasoning, Qwen for coding).
5. **Model Serving & Fallback:**
   - *Responsibility:* Ensure high availability of LLM inference.
   - *Tech Choice:* NVIDIA NIM primary, Groq/Gemini fallbacks with Tenacity retries.
   - *Failure Mode:* Rate limits. *Mitigation:* Custom sliding-window rate limiter shifting traffic to fallbacks.

### Section 4 — Data Flow (Step-by-Step)
```
Step 1: [Raw User String] → [FastAPI /chat/stream] → [LangGraph Orchestrator]
Step 2: [Thread ID] → [SqliteSaver] → [Retrieves past BaseMessages]
Step 3: [Full Context] → [safety_agent.py] → [Checks for malicious intent]
Step 4: [Verified Context] → [router.py] → [Selects specialized agent]
Step 5: [Context] → [Specialized Agent (e.g., coding_agent.py)] → [Fallback Handler]
Step 6: [Prompt] → [NVIDIA NIM / Groq] → [Returns token chunks]
Step 7: [Token Chunks] → [FastAPI StreamingResponse] → [Streamlit Frontend]
Step 8: [Final AIMessage] → [SqliteSaver] → [Persisted to DB]
```

### Section 5 — Scalability Plan

| Scale | Architecture Change |
|-------|-------------------|
| 100 users | Current SQLite + FastAPI works. |
| 10K users | Replace SQLite with PostgreSQL (`PostgresSaver`). Add Redis for rate-limiter state. |
| 100K users | Horizontally scale FastAPI across multiple pods in K8s. Implement PgBouncer. |
| 1M users | Add Redis Semantic Cache to bypass LLM for frequent identical queries. |

*Stateless vs Stateful:* The FastAPI backend is entirely stateless, making it trivial to scale horizontally. The state is pushed to the database (SQLite currently, Postgres eventually).

### Section 6 — Performance Optimization
- **Latency:** Implementing Server-Sent Events (SSE) brings TTFT (Time To First Token) to <500ms, improving perceived performance.
- **Dynamic Batching:** Not applicable here as we rely on third-party APIs (NVIDIA NIM).
- **Rate Limiting:** The local rate limiter intercepts requests *before* hitting 429s, seamlessly shifting traffic to Groq, eliminating dead time waiting for external API resets.

### Section 7 — Reliability & Fault Tolerance
- **Scenario:** Primary LLM API (NVIDIA) is down.
  - *Detection:* `openai.RateLimitError` or 5xx.
  - *Recovery:* `Tenacity` triggers `models/fallback.py`, which consults the `RateLimiter` and routes to Groq or Gemini.
- **Scenario:** SQLite Lock Exception.
  - *Detection:* `OperationalError` during checkpoint write.
  - *Recovery:* Need to implement DB retry logic, or migrate to Postgres.
- **Scenario:** Safety Agent failure.
  - *Detection:* Exception in `check_safety`.
  - *Recovery:* Designed to fail-open ("check skipped") to ensure application availability over strict compliance.

### Section 8 — Cost Optimization Analysis
| Component | Current Cost Driver | Optimization Strategy |
|-----------|--------------------|-----------------------|
| Compute | LLM API Token usage | Route simple chats to cheaper/free models (Llama 8B) instead of 70B models. |
| Infrastructure| Running containers 24/7 | Deploy FastAPI on serverless (e.g., Cloud Run/AWS App Runner) scaling to zero. |

### Section 9 — Observability Stack (CRITICAL)
- **Metrics to track:**
  - Token usage per agent/model (Cost attribution).
  - TTFT and end-to-end latency.
  - Fallback trigger rate (how often are we failing over to Groq?).
- **Implementation:** LangSmith is critical here for tracking agent trajectories and identifying routing errors in `router.py`.

### Section 10 — System Design Interview Questions
1. **Q:** Why did you decouple Streamlit and FastAPI instead of using Streamlit's built-in session state for everything?
   **A:** Streamlit's execution model reruns the entire script on interaction. Decoupling allows the FastAPI backend to scale independently, prevents memory leaks in the UI thread, and allows other clients (like a mobile app) to consume the same LangGraph state.
2. **Q:** How does your fallback rate limiter behave in a multi-worker environment (e.g., `uvicorn --workers 4`)?
   **A:** Currently, it uses a local `threading.Lock`, which fails across workers because memory isn't shared. To fix this at scale, the state must be moved to Redis using atomic `INCR` and `EXPIRE` commands.
