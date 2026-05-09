import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import uuid
import requests
import json
import os
import base64

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

# ── Helpers ───────────────────────────────────────────────────────────────────

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

if st.sidebar.button("＋ New Chat", type="primary"):
    reset_chat()
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

# ── Chat display ──────────────────────────────────────────────────────────────

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
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
}

def run_stream(payload, status, message_placeholder):
    full_response = ""
    first_token = True
    first_visible_node = True
    interrupted = False
    tool_calls = None

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
                                    if first_visible_node:
                                        status.update(label="⚙️ Orchestrating...", state="running")
                                        first_visible_node = False
                                    status.write(f"✅ {NODE_LABELS[node_name]}")

                            elif event_type == "token":
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
        message_placeholder.markdown(full_response)
        return full_response, interrupted, tool_calls

    except Exception as e:
        status.update(label="❌ Failed", state="error", expanded=False)
        st.error(f"Error communicating with backend: {e}")
        full_response = "Sorry, I am unable to connect to the backend server right now."
        message_placeholder.markdown(full_response)
        return full_response, False, None

# ── Resume Interceptor ────────────────────────────────────────────────────────
if "resume_action" in st.session_state:
    action = st.session_state.pop("resume_action")
    with st.chat_message("assistant"):
        status = st.status(f"⏳ {'Approving' if action == 'approve' else 'Rejecting'} action...", expanded=True)
        message_placeholder = st.empty()
        
        payload = {"message": "", "thread_id": st.session_state["thread_id"], "action": action}
        full_response, interrupted, tool_calls = run_stream(payload, status, message_placeholder)
        
        msg_data = {"role": "assistant", "content": full_response}
        if interrupted:
            msg_data["interrupt"] = tool_calls
            
        st.session_state["message_history"].append(msg_data)
    st.rerun()

# ── Chat input ─────────────────────────────────────────────────────────────
user_input = st.chat_input("Type here")

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

        full_response, interrupted, tool_calls = run_stream(payload, status, message_placeholder)

        msg_data = {"role": "assistant", "content": full_response}
        if interrupted:
            msg_data["interrupt"] = tool_calls
            
        st.session_state["message_history"].append(msg_data)

    refresh_title(st.session_state["thread_id"])
    st.rerun()
