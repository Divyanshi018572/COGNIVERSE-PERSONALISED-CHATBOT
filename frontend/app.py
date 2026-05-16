import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import streamlit as st
import uuid
import requests
import json
import os
import base64
import re
import streamlit.components.v1 as components

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ── Session init ──────────────────────────────────────────────────────────────

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []

if "thread_titles" not in st.session_state:
    st.session_state["thread_titles"] = {}

if "show_uploader" not in st.session_state:
    st.session_state["show_uploader"] = False

if "attached_file_name" not in st.session_state:
    st.session_state["attached_file_name"] = None

if "agent_trace" not in st.session_state:
    st.session_state["agent_trace"] = []

if "performance_stats" not in st.session_state:
    st.session_state["performance_stats"] = {"latency": 0.0, "tokens": 0, "tps": 0.0}

# ── Helpers ───────────────────────────────────────────────────────────────────

NODE_LABELS = {
    "router":           "🧭 Routing to Best Agent",
    "chat_agent":       "💬 Chat Agent",
    "reasoning_agent":  "🧠 Reasoning Agent",
    "coding_agent":     "💻 Coding Agent",
    "coding_tools":     "🛠️ Executing Tools",
    "research_agent":   "🔍 Research Agent",
    "rag_agent":        "📚 RAG Document Agent",
    "vision_agent":     "👁️ Vision Agent",
    "memory_agent":     "💾 Extracting Memories",
    "github_agent":     "🐙 GitHub Architecture Agent",
    "evaluator":        "⚖️ Quality Audit (High-Res)",
}

def get_score_color(score: float) -> str:
    if score >= 0.9: return "green"
    if score >= 0.7: return "orange"
    return "red"

def make_request(method, endpoint, **kwargs):
    url = f"{BACKEND_URL}{endpoint}"
    return requests.request(method, url, **kwargs)

def generate_thread_id() -> str:
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(st.session_state["thread_id"])
    st.session_state["message_history"] = []

def add_thread(thread_id: str):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def load_conversation(thread_id: str):
    try:
        response = make_request("POST", "/thread/history", json={"thread_id": thread_id})
        if response.status_code == 200:
            return response.json().get("messages", [])
    except Exception as e:
        st.error(f"Error loading conversation: {e}")
    return []

def fetch_threads():
    try:
        response = make_request("GET", "/threads")
        if response.status_code == 200:
            return response.json().get("threads", [])
    except Exception:
        pass
    return []

def get_feedback_stats():
    try:
        response = make_request("GET", "/feedback/stats")
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return {"total": 0, "positive": 0, "negative": 0, "avg_eval_score": 0.0, "satisfaction_rate": 0.0}

def delete_thread(thread_id: str) -> bool:
    try:
        r = make_request("DELETE", f"/thread/{thread_id}", timeout=10)
        return r.status_code == 200 and r.json().get("success", False)
    except Exception:
        return False

def refresh_title(thread_id: str):
    try:
        r = make_request("POST", "/title", json={"thread_id": thread_id}, timeout=15)
        if r.status_code == 200:
            title = r.json().get("title", "")
            if title:
                st.session_state.setdefault("thread_titles", {})[thread_id] = title
    except Exception:
        pass

def get_thread_title(thread_id: str) -> str:
    if "thread_titles" not in st.session_state:
        st.session_state["thread_titles"] = {}
        
    if thread_id not in st.session_state["thread_titles"]:
        refresh_title(thread_id)
        
    titles = st.session_state.get("thread_titles", {})
    return titles.get(thread_id, f"• {thread_id[:8]}...")

# ── Load Initialization ──────────────────────────────────────────────────────
if not st.session_state["chat_threads"]:
    st.session_state["chat_threads"] = fetch_threads()
    if st.session_state["chat_threads"]:
        st.session_state["thread_id"] = st.session_state["chat_threads"][-1]
        st.session_state["message_history"] = load_conversation(st.session_state["thread_id"])

# ── Main Chat Interface ──────────────────────────────────────────────────────

add_thread(st.session_state["thread_id"])

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("Cognibot 🧠")

# ── Page Navigation ───────────────────────────────────────────────────────────
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Overview"

page = st.sidebar.radio(
    "Navigate",
    ["💬 Chat", "🌐 Overview"],
    index=0 if st.session_state["current_page"] == "Chat" else 1,
    label_visibility="collapsed"
)
st.session_state["current_page"] = "Chat" if page == "💬 Chat" else "Overview"


stats = get_feedback_stats()
if stats["total"] > 0:
    st.sidebar.divider()
    st.sidebar.subheader("📊 Quality Metrics")
    colA, colB = st.sidebar.columns(2)
    with colA:
        st.metric("Satisfaction", f"{stats['satisfaction_rate']}%")
    with colB:
        st.metric("Avg Eval Score", f"{stats['avg_eval_score']}")
    st.sidebar.caption(f"Based on {stats['total']} responses ({stats['positive']} 👍 / {stats['negative']} 👎)")

st.sidebar.divider()
st.sidebar.subheader("📡 Agentic Pulse")

# Live Trace Monitor
if st.session_state["agent_trace"]:
    for step in st.session_state["agent_trace"][-5:]: # Show last 5 hops
        status_icon = "🔵" if step["status"] == "active" else "✅"
        st.sidebar.markdown(f"{status_icon} **{step['label']}**")
        st.sidebar.caption(f"Time: {step['time']}")
else:
    st.sidebar.info("Waiting for agent activation...")

st.sidebar.divider()
st.sidebar.subheader("🚀 Performance HUD")
pstats = st.session_state["performance_stats"]
st.sidebar.markdown(f"**Latency:** `{pstats['latency']:.2f}s`")
st.sidebar.markdown(f"**Throughput:** `{pstats['tps']:.1f}` tok/s")

st.sidebar.divider()

if st.sidebar.button("＋ New Chat", type="primary"):
    reset_chat()
    st.rerun()

st.sidebar.divider()

with st.sidebar.expander("🐙 GitHub Integration", expanded=False):
    repo_url = st.text_input("Repository URL", placeholder="https://github.com/user/repo")
    if st.button("Analyze Repo", use_container_width=True):
        if repo_url:
            st.session_state["auto_send"] = f"Analyze the GitHub repository: {repo_url}. Provide the architecture flowchart, tech stack visuals, suggest how to contribute, and provide a command to clone it."
            st.rerun()
            
    st.divider()
    topic = st.text_input("Topic Search", placeholder="e.g. machine-learning")
    if st.button("Find Top Repos", use_container_width=True):
        if topic:
            st.session_state["auto_send"] = f"Find top repositories for the topic '{topic}' focusing on content, commits, and issues. Present a comparison table."
            st.rerun()

st.sidebar.divider()
if st.sidebar.button("📎 Attach File"):
    st.session_state["show_uploader"] = not st.session_state["show_uploader"]
    st.rerun()

IMAGE_TYPES   = ["png", "jpg", "jpeg", "webp", "gif"]
DOC_TYPES     = ["csv", "md", "txt", "pdf", "xlsx", "xls"]
ALL_TYPES     = IMAGE_TYPES + DOC_TYPES

file_bytes_b64 = None   # image path: base64 for vision agent
file_type      = None   # image mime type
file_name      = None   # doc path: filename passed to backend for RAG

if st.session_state["show_uploader"]:
    uploaded_file = st.sidebar.file_uploader(
        "Choose a file",
        type=ALL_TYPES,
        help="Images: PNG, JPG, GIF, WEBP · Docs: CSV, TXT, MD, PDF, XLSX"
    )
    if uploaded_file is not None:
        fname = uploaded_file.name
        ext   = fname.rsplit(".", 1)[-1].lower()
        raw_bytes = uploaded_file.read()

        if ext in IMAGE_TYPES:
            # Images → base64 for the vision agent (no ingestion needed)
            file_bytes_b64 = base64.b64encode(raw_bytes).decode("utf-8")
            file_type = uploaded_file.type
            st.sidebar.success(f"✅ Image attached: {fname}")

        else:
            # Documents → ingest into ChromaDB via backend /ingest endpoint
            with st.sidebar:
                with st.spinner(f"📥 Ingesting {fname} into knowledge base…"):
                    try:
                        b64 = base64.b64encode(raw_bytes).decode("utf-8")
                        resp = make_request(
                            "POST", "/ingest",
                            json={
                                "thread_id":     st.session_state["thread_id"],
                                "file_bytes_b64": b64,
                                "file_name":      fname,
                            },
                            timeout=60,
                        )
                        result = resp.json()
                        if result.get("success"):
                            chunks = result.get("chunks", "?")
                            st.sidebar.success(
                                f"✅ **{fname}** ingested into RAG "
                                f"({chunks} chunks). Ask me anything about it!"
                            )
                            file_name = fname   # signal to chat payload
                            st.session_state["attached_file_name"] = fname
                        else:
                            st.sidebar.error(
                                f"❌ Ingestion failed: {result.get('error', 'Unknown error')}"
                            )
                    except Exception as e:
                        st.sidebar.error(f"❌ Could not reach backend: {e}")


st.sidebar.divider()
st.sidebar.header("My Conversations")

for tid in st.session_state["chat_threads"][::-1]:
    title = get_thread_title(tid)
    is_active = tid == st.session_state["thread_id"]
    btn_label = f"**· {title}**" if is_active else title

    col_del, col_title = st.sidebar.columns([1, 5])

    if col_del.button("🗑️", key=f"del_{tid}", help=f"Delete this conversation"):
        with st.sidebar:
            with st.spinner("Deleting…"):
                ok = delete_thread(tid)
        if ok:
            st.session_state["chat_threads"] = [
                t for t in st.session_state["chat_threads"] if t != tid
            ]
            st.session_state.get("thread_titles", {}).pop(tid, None)
            if st.session_state["thread_id"] == tid:
                new_tid = generate_thread_id()
                st.session_state["thread_id"] = new_tid
                st.session_state["message_history"] = []
                st.session_state["chat_threads"].append(new_tid)
            st.toast(f"🗑️ Conversation deleted", icon="✅")
        else:
            st.sidebar.error("Failed to delete.")
        st.rerun()

    if col_title.button(btn_label, key=f"thread_{tid}", use_container_width=True):
        st.session_state["thread_id"] = tid
        st.session_state["message_history"] = load_conversation(tid)
        st.rerun()

# ── Overview Page ─────────────────────────────────────────────────────────────

def render_overview():
    st.markdown("""
    <style>
    .ov-hero { 
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        border-radius: 16px; padding: 48px 40px; text-align: center; margin-bottom: 32px;
    }
    .ov-hero h1 { font-size: 2.8rem; font-weight: 800; color: #fff; margin: 0 0 10px; }
    .ov-hero p  { font-size: 1.1rem; color: #a0aec0; max-width: 680px; margin: 0 auto; }
    .feat-card {
        background: #161b22; border: 1px solid #30363d; border-radius: 12px;
        padding: 20px 22px; height: 100%; transition: border-color 0.2s;
    }
    .feat-card:hover { border-color: #58a6ff; }
    .feat-icon { font-size: 2rem; margin-bottom: 8px; }
    .feat-title { font-size: 1rem; font-weight: 700; color: #e6edf3; margin-bottom: 6px; }
    .feat-desc  { font-size: 0.85rem; color: #8b949e; line-height: 1.5; }
    .agent-badge {
        display: inline-block; background: #21262d; border: 1px solid #30363d;
        border-radius: 20px; padding: 4px 14px; font-size: 0.82rem; color: #58a6ff;
        margin: 4px 3px;
    }
    .stack-pill {
        display: inline-block; background: #0d1117; border: 1px solid #30363d;
        border-radius: 6px; padding: 3px 10px; font-size: 0.78rem; color: #7ee787;
        margin: 3px 2px; font-family: monospace;
    }
    .revert-box {
        background: #161b22; border: 1px solid #f0883e; border-radius: 12px;
        padding: 20px 24px; margin-top: 24px;
    }
    .revert-box h4 { color: #f0883e; margin: 0 0 8px; }
    .revert-box code { background: #0d1117; border-radius: 4px; padding: 2px 6px; color: #a5d6ff; font-size: 0.9rem; }
    </style>
    """, unsafe_allow_html=True)

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="ov-hero">
        <h1>🧠 Cognibot — COGNIVERSE</h1>
        <p>A production-grade, multi-agent AI assistant powered by LangGraph,
           NVIDIA NIM, and Groq. Routes every query to the right specialist agent
           in real-time with live streaming, semantic caching, and HITL oversight.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Core Feature Cards ────────────────────────────────────────────────────
    st.subheader("⚡ Core Features")
    c1, c2, c3 = st.columns(3)
    cards = [
        ("🤖", "Multi-Agent Orchestration",
         "LangGraph routes queries to 9 specialist agents: Chat, Reasoning, Coding, Research, RAG, Vision, GitHub, Memory, and Evaluator."),
        ("🌊", "Real-Time Token Streaming",
         "Responses stream token-by-token via SSE. The Agentic Pulse sidebar shows live agent hops and execution steps."),
        ("📚", "RAG Document Pipeline",
         "Upload PDFs, CSVs, DOCX, or TXT files. ChromaDB + BM25 hybrid retrieval with NVIDIA re-ranking gives context-aware answers."),
        ("👁️", "Vision Agent",
         "Upload images (PNG, JPG, GIF, WEBP) and ask questions. Powered by meta/llama-3.2-11b-vision-instruct via NVIDIA NIM."),
        ("🔍", "Live Web Research",
         "The Research Agent uses Tavily Search + ArXiv to fetch real-time information and up-to-date academic papers."),
        ("🐙", "GitHub Integration",
         "Paste any public GitHub repo URL. The agent generates architecture diagrams, tech stack analysis, and contribution guides."),
        ("⚖️", "AI Quality Auditor",
         "Every response is evaluated by a High-Res Auditor agent that scores confidence and flags low-quality answers for improvement."),
        ("💾", "Persistent Memory",
         "Conversations are stored in PostgreSQL. The Memory Agent extracts long-term facts and the History API restores past sessions."),
        ("🛡️", "Safety Guard",
         "A dedicated Safety Agent screens every message before routing, blocking harmful or off-policy requests automatically."),
    ]
    all_cols = [c1, c2, c3]
    for idx, (icon, title, desc) in enumerate(cards):
        with all_cols[idx % 3]:
            st.markdown(f"""
            <div class="feat-card">
                <div class="feat-icon">{icon}</div>
                <div class="feat-title">{title}</div>
                <div class="feat-desc">{desc}</div>
            </div><br/>
            """, unsafe_allow_html=True)

    # ── Agent Roster ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🗂️ Agent Roster")
    agents = [
        ("🧭", "Router",          "Semantic router — classifies intent and dispatches to correct agent"),
        ("💬", "Chat Agent",      "General Q&A, summaries, explanations"),
        ("🧠", "Reasoning Agent", "Step-by-step logic, math, complex analysis"),
        ("💻", "Coding Agent",    "Code gen, debugging, Python REPL execution"),
        ("🔍", "Research Agent",  "Tavily + ArXiv live web & paper search"),
        ("📚", "RAG Agent",       "Document Q&A over ingested files via ChromaDB"),
        ("👁️", "Vision Agent",   "Image understanding & visual Q&A"),
        ("🐙", "GitHub Agent",    "Repo analysis, architecture diagrams, contributor tips"),
        ("💾", "Memory Agent",    "Long-term fact extraction & session persistence"),
        ("⚖️", "Evaluator",      "Response quality audit with confidence scoring"),
        ("🛡️", "Safety Agent",   "Pre-routing content policy enforcement"),
    ]
    for icon, name, role in agents:
        col_a, col_b = st.columns([1, 5])
        with col_a:
            st.markdown(f"**{icon} {name}**")
        with col_b:
            st.caption(role)

    # ── Tech Stack ────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🏗️ Tech Stack")
    stack = {
        "🤖 AI / Models": ["LangGraph", "LangChain", "NVIDIA NIM", "Groq", "Google Gemini", "Llama 3.3", "DeepSeek R1"],
        "🖥️ Backend":     ["FastAPI", "Uvicorn", "SSE Streaming", "PostgreSQL", "Redis Stack", "ChromaDB"],
        "🎨 Frontend":    ["Streamlit", "Mermaid.js", "Custom CSS", "Base64 Vision"],
        "🔧 DevOps":      ["Docker", "docker-compose", "python-dotenv", "Tenacity", "structlog"],
    }
    for group, items in stack.items():
        st.markdown(f"**{group}**")
        pills_html = "".join(f'<span class="stack-pill">{i}</span>' for i in items)
        st.markdown(f'<div style="margin-bottom:12px">{pills_html}</div>', unsafe_allow_html=True)



# ── Chat display ──────────────────────────────────────────────────────────────

def render_feedback_buttons(question: str, answer: str, msg_id: str):

    col1, col2, col3 = st.columns([1, 1, 8])
    feedback_key = f"feedback_{msg_id}"
    
    if feedback_key not in st.session_state:
        with col1:
            if st.button("👍", key=f"up_{feedback_key}"):
                make_request("POST", "/feedback", json={
                    "thread_id": st.session_state["thread_id"],
                    "question": question,
                    "answer": answer,
                    "thumbs_up": True
                })
                st.session_state[feedback_key] = "positive"
                st.rerun()
        with col2:
            if st.button("👎", key=f"down_{feedback_key}"):
                st.session_state[feedback_key] = "negative_pending"
                st.rerun()
                
    if st.session_state.get(feedback_key) == "negative_pending":
        comment = st.text_input("What went wrong?", key=f"comment_{feedback_key}")
        if st.button("Submit Feedback", key=f"submit_{feedback_key}"):
            make_request("POST", "/feedback", json={
                "thread_id": st.session_state["thread_id"],
                "question": question,
                "answer": answer,
                "thumbs_up": False,
                "comment": comment
            })
            st.session_state[feedback_key] = "negative"
            st.session_state["auto_send"] = f"Please improve your previous response based on this feedback: {comment}"
            st.rerun()
            
    elif st.session_state.get(feedback_key) in ["positive", "negative"]:
        with col1:
            st.caption("✅ Feedback recorded")


def render_message_with_mermaid(content: str):
    """Parses markdown content and safely renders Mermaid blocks via HTML components."""
    parts = re.split(r'```mermaid\n(.*?)\n```', content, flags=re.DOTALL)
    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part.strip():
                st.markdown(part)
        else:
            # Safely encode the mermaid string for javascript
            clean_part = part.strip()
            
            # AI models often hallucinate `-->|text|>` instead of `-->|text|`. Fix it automatically.
            clean_part = re.sub(r'\|([^|]+)\|>', r'|\1|', clean_part)
            
            safe_code = json.dumps(clean_part)
            
            mermaid_html = f"""<!DOCTYPE html>
<html>
<head>
    <style>
        html, body {{
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            background-color: transparent;
            overflow: hidden;
            color: white;
            font-family: sans-serif;
        }}
        #output {{
            width: 100%;
            height: 100%;
            display: flex;
            justify-content: center;
            align-items: center;
            background-color: #0e1117;
            border-radius: 8px;
            overflow: hidden;
        }}
        #output svg {{
            max-width: 100% !important;
            max-height: 100% !important;
            width: auto !important;
            height: auto !important;
        }}
    </style>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ 
            startOnLoad: false, 
            theme: 'base',
            themeVariables: {{
                primaryColor: '#ff4b4b',
                primaryTextColor: '#fff',
                primaryBorderColor: '#ff4b4b',
                lineColor: '#6e7681',
                secondaryColor: '#1f2937',
                tertiaryColor: '#111827',
                mainBkg: '#0e1117',
                nodeBorder: '#30363d',
                clusterBkg: '#161b22',
                clusterBorder: '#30363d',
                defaultLinkColor: '#8b949e',
                titleColor: '#58a6ff',
                edgeLabelBackground:'#1f2937'
            }},
            securityLevel: 'loose',
            flowchart: {{ 
                useMaxWidth: true, 
                htmlLabels: true,
                curve: 'basis'
            }}
        }});
        
        async function renderDiagram() {{
            const code = {safe_code};
            try {{
                const {{ svg }} = await mermaid.render('mermaid-svg', code);
                document.getElementById('output').innerHTML = svg;
            }} catch (err) {{
                document.getElementById('output').innerHTML = '<span style="color:red">Mermaid Syntax Error</span><br><pre style="color:red;font-size:10px">' + err + '</pre><pre style="font-size:10px;color:white;">' + code + '</pre>';
            }}
        }}
        window.addEventListener('load', renderDiagram);
    </script>
</head>
<body>
    <div id="output">Rendering diagram...</div>
</body>
</html>"""
            st.components.v1.html(mermaid_html, height=500, scrolling=False)



for i, message in enumerate(st.session_state["message_history"]):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_message_with_mermaid(message["content"])
            
            # Persistent Confidence Badge
            if "score" in message:
                score = message["score"]
                color = get_score_color(score)
                badge_html = f"""
                <div style="display: flex; align-items: center; gap: 8px; margin-top: 10px; padding: 4px 12px; background-color: #161b22; border-radius: 16px; border: 1px solid #30363d; width: fit-content;">
                    <span style="font-size: 14px;">🛡️ Auditor Confidence:</span>
                    <b style="color: {color}; font-size: 14px;">{int(score*100)}%</b>
                </div>
                """
                st.markdown(badge_html, unsafe_allow_html=True)
                if score < 0.9 and "critique" in message:
                    st.caption(f"📝 **Auditor Note:** {message['critique']}")
        else:
            st.markdown(message["content"])
        
        # Render feedback buttons for AI messages (excluding if interrupted)
        if message["role"] == "assistant" and "interrupt" not in message:
            # Get the previous human message as the question
            prev_msg = st.session_state["message_history"][i-1] if i > 0 else None
            question = prev_msg["content"] if prev_msg and prev_msg["role"] == "user" else "N/A"
            msg_id = message.get("id", f"{i}_{hash(message['content'])}")
            render_feedback_buttons(question, message["content"], msg_id)
        
        if "interrupt" in message:
            st.warning("⚠️ **Approval Required:** The agent wants to execute the following actions:")
            for idx, tc in enumerate(message["interrupt"]):
                with st.expander(f"Tool: `{tc['name']}`", expanded=True):
                    if tc['name'] == 'Python_REPL' and 'query' in tc['args']:
                        st.code(tc['args']['query'], language='python')
                    elif tc['name'] == 'write_file' and 'file_path' in tc['args'] and 'text' in tc['args']:
                        st.markdown(f"**Target File:** `{tc['args']['file_path']}`")
                        st.code(tc['args']['text'], language='python')
                    else:
                        st.code(json.dumps(tc["args"], indent=2))
                    
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Approve", key=f"approve_{message.get('id', id(message))}"):
                    st.session_state["resume_action"] = "approve"
                    del message["interrupt"]
                    st.rerun()
            with col2:
                if st.button("❌ Reject", key=f"reject_{message.get('id', id(message))}"):
                    st.session_state["resume_action"] = "reject"
                    del message["interrupt"]
                    st.rerun()


def run_stream(payload, status, message_placeholder):
    full_response = ""
    first_token = True
    first_visible_node = True
    interrupted = False
    tool_calls = None
    current_score = None
    current_critique = None
    
    # Dashboard Tracking
    st.session_state["agent_trace"] = [] # Reset for new turn
    start_time = time.time()
    token_count = 0

    try:
        with requests.post(
            f"{BACKEND_URL}/chat/stream",
            json=payload,
            stream=True
        ) as r:
            for line in r.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8")
                    if decoded_line.startswith("data: "):
                        data_str = decoded_line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            event_type = data.get("type", "token")

                            if event_type == "node":
                                node_name = data.get("node", "")
                                if node_name == "safety":
                                    pass
                                elif node_name == "blocked":
                                    reason = data.get("reason", "Content policy violation")
                                    status.update(label="🚫 Unsafe Content", state="error")
                                    status.write(f":red[**{reason}**]")
                                elif node_name in NODE_LABELS:
                                    label = NODE_LABELS[node_name]
                                    # Update Live Trace
                                    st.session_state["agent_trace"].append({
                                        "node": node_name,
                                        "label": label,
                                        "time": time.strftime("%H:%M:%S"),
                                        "status": "active"
                                    })
                                    if first_visible_node:
                                        status.update(label=f"⚙️ {label} Active...", state="running")
                                        first_visible_node = False
                                    else:
                                        status.write(f"✅ {label} Complete")
                                    
                                    label = NODE_LABELS[node_name]
                                    if node_name == "evaluator":
                                        current_score = data.get("score", 0.0)
                                        current_critique = data.get("critique", "")
                                        color = get_score_color(current_score)
                                        label = f"{label} - :{color}[**{int(current_score*100)}% Confidence**]"
                                        status.write(f"✅ {label}")
                                        if current_score < 0.9:
                                            with status:
                                                st.caption(f"Critique: {current_critique}")
                                    else:
                                        status.write(f"✅ {label}")

                            elif event_type == "token":
                                token_count += 1
                                if first_token:
                                    status.update(label="✍️ Generating response...", state="running")
                                    first_token = False
                                full_response += data.get("content", "")
                                message_placeholder.markdown(full_response + "▌")

                            elif event_type == "interrupt":
                                interrupted = True
                                tool_calls = data.get("tool_calls", [])
                                status.update(label="⚠️ Approval Required", state="error")
                                status.write(":orange[**The agent has paused to request permission.**]")

                            elif event_type == "error":
                                status.write(f":red[❌ {data.get('content', 'Unknown error')}]")

                        except json.JSONDecodeError:
                            pass

        if not interrupted:
            status.update(label="✅ Done", state="complete", expanded=False)
        # Re-render the final response using our mermaid helper instead of simple markdown
        message_placeholder.empty() # Clear the streaming placeholder
        with message_placeholder.container():
            render_message_with_mermaid(full_response)
            
        # Finalize Dashboard Stats
        end_time = time.time()
        duration = end_time - start_time
        tps = token_count / duration if duration > 0 else 0
        st.session_state["performance_stats"] = {
            "latency": duration,
            "tokens": token_count,
            "tps": tps
        }
        # Mark all as complete
        for step in st.session_state["agent_trace"]:
            step["status"] = "complete"

        return full_response, interrupted, tool_calls, current_score, current_critique

    except Exception as e:
        status.update(label="❌ Failed", state="error", expanded=False)
        st.error(f"Error communicating with backend: {e}")
        full_response = "Sorry, I am unable to connect to the backend server right now."
        with message_placeholder.container():
            render_message_with_mermaid(full_response)
        return full_response, False, None, None, None

# ── Page Router ───────────────────────────────────────────────────────────────
if st.session_state.get("current_page") == "Overview":
    render_overview()
    st.stop()

# ── Resume Interceptor ────────────────────────────────────────────────────────
if "resume_action" in st.session_state:

    action = st.session_state.pop("resume_action")
    with st.chat_message("assistant"):
        status = st.status(f"⏳ {'Approving' if action == 'approve' else 'Rejecting'} action...", expanded=True)
        message_placeholder = st.empty()
        
        payload = {"message": "", "thread_id": st.session_state["thread_id"], "action": action}
        full_response, interrupted, tool_calls, score, critique = run_stream(payload, status, message_placeholder)
        
        msg_data = {"role": "assistant", "content": full_response}
        if interrupted:
            msg_data["interrupt"] = tool_calls
        if score is not None:
            msg_data["score"] = score
            msg_data["critique"] = critique
            
        st.session_state["message_history"].append(msg_data)
    st.rerun()

# ── Chat input ─────────────────────────────────────────────────────────────
user_input = st.chat_input("Type here")
auto_send = st.session_state.pop("auto_send", None)

if auto_send:
    user_input = auto_send

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        status = st.status("⏳ Checking...", expanded=True)
        message_placeholder = st.empty()

        payload = {"message": user_input, "thread_id": st.session_state["thread_id"]}
        if file_bytes_b64 and file_type:
            # Image → vision agent via base64
            payload["file_bytes"] = file_bytes_b64
            payload["file_type"] = file_type
        elif file_name:
            # Document was ingested into ChromaDB; just pass the filename so
            # the backend sets task='rag' and routes to the RAG agent.
            payload["file_name"] = file_name

        full_response, interrupted, tool_calls, score, critique = run_stream(payload, status, message_placeholder)

        msg_data = {"role": "assistant", "content": full_response}
        if interrupted:
            msg_data["interrupt"] = tool_calls
        if score is not None:
            msg_data["score"] = score
            msg_data["critique"] = critique
            
        st.session_state["message_history"].append(msg_data)

    refresh_title(st.session_state["thread_id"])
    st.rerun()
