from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import json
import os
import base64
from langchain_core.messages import HumanMessage, AIMessage

from core.orchestrator import get_orchestrator, pool
from core.db import init_db
from rag.document_processor import process_document
from rag.vector_store import ingest_documents, query_documents

from contextlib import asynccontextmanager
from langchain_core.globals import set_llm_cache
from langchain_community.cache import RedisSemanticCache
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from utils.logger import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    try:
        from langchain_community.cache import InMemoryCache
        set_llm_cache(InMemoryCache())
        logger.info("In-Memory LLM Cache Initialized (RedisSemanticCache skipped - redis version incompatibility)")
    except Exception as e:
        logger.error(f"Failed to initialize LLM Cache: {e}")
    yield

app = FastAPI(title="Cognibot Multi-Agent API", lifespan=lifespan)

# --- WebMCP Discovery ---
@app.get("/mcp-actions.json")
async def mcp_discovery():
    """Publishes machine-readable actions for AI browsing agents."""
    return {
        "version": "1.0",
        "site": "http://localhost:8501",
        "actions": [
            {
                "id": "analyze-github-repo",
                "name": "Analyze GitHub Repository",
                "description": "Analyzes a GitHub repository's architecture, tech stack, and contribution guide.",
                "method": "declarative",
                "endpoint": "/chat",
                "parameters": {
                    "required": ["message"],
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "A message containing the GitHub URL to analyze, e.g., 'Analyze https://github.com/langchain-ai/langchain'"
                        }
                    }
                }
            }
        ]
    }

# Auth router removed

from typing import Optional

class ChatRequest(BaseModel):
    message: str = Field(..., max_length=4000)
    thread_id: str
    file_bytes: Optional[str] = None
    file_type: Optional[str] = None
    file_content: Optional[str] = None   # extracted text from doc uploads (legacy fallback)
    file_name: Optional[str] = None      # original filename for context
    action: Optional[str] = None

class IngestRequest(BaseModel):
    thread_id: str
    file_bytes_b64: str          # base64-encoded raw file bytes
    file_name: str               # original filename (used to detect extension)

class ThreadRequest(BaseModel):
    thread_id: str

@app.post("/chat/stream")
def chat_stream(request: ChatRequest, user_id: int = 1):
    """Sync endpoint so the sync LangGraph iterator streams chunks immediately."""
    orchestrator = get_orchestrator()
    config = {"configurable": {"thread_id": request.thread_id, "user_id": user_id}}

    if request.file_bytes and request.file_type:
        # Image upload — use multimodal vision format; router will pick vision_agent
        content = [
            {"type": "text", "text": request.message},
            {"type": "image_url", "image_url": {"url": f"data:{request.file_type};base64,{request.file_bytes}"}}
        ]
        msg = HumanMessage(content=content)
    elif request.file_name:
        # Document was uploaded: the ingest endpoint already pushed it into ChromaDB.
        # Send a plain message; the router will detect task='rag' via the state override below.
        msg = HumanMessage(content=request.message)
    elif request.file_content:
        # Legacy fallback: frontend sent extracted text directly
        file_label = f"[File: {request.file_name}]" if request.file_name else "[Attached file]"
        combined = (
            f"{file_label}\n"
            f"```\n{request.file_content[:12000]}\n```\n\n"
            f"{request.message}"
        )
        msg = HumanMessage(content=combined)
    else:
        msg = HumanMessage(content=request.message)

    # Force 'rag' task and set file_path when a document filename is present
    initial_state_override = {"task": "rag", "file_path": request.file_name} if request.file_name else {}

    def generate():
        streamed_nodes = set()
        try:
            if request.action == "reject":
                from langchain_core.messages import ToolMessage
                state = orchestrator.get_state(config)
                last_msg = state.values["messages"][-1]
                tool_msgs = []
                for tc in last_msg.tool_calls:
                    tool_msgs.append(ToolMessage(content="User rejected the tool execution.", tool_call_id=tc["id"]))
                orchestrator.update_state(config, {"messages": tool_msgs}, as_node="coding_tools")
                stream_input = None
            elif request.action == "approve":
                stream_input = None
            else:
                stream_input = {"messages": [msg], "retry_count": 0, **initial_state_override}
                
            for event in orchestrator.stream(
                stream_input,
                config=config,
                stream_mode=["messages", "updates"]
            ):
                mode = event[0]
                payload = event[1]

                if mode == "updates":
                    for node_name, state_update in payload.items():
                        data = {"type": "node", "node": node_name}
                        
                        # Pass block reason for unsafe content
                        if node_name == "blocked":
                            msgs = state_update.get("messages", [])
                            if msgs:
                                data["reason"] = msgs[-1].content
                        
                        # Pass evaluation score for the current turn
                        if node_name == "evaluator":
                            data["score"] = state_update.get("eval_score", 0.0)
                            data["critique"] = state_update.get("eval_feedback", "")
                                
                        # Fallback for models/wrappers that completely fail to use stream callbacks
                        if node_name not in streamed_nodes and node_name not in ["safety", "router", "evaluator", "memory_agent", "coding_tools", "blocked"]:
                            msgs = state_update.get("messages", [])
                            if msgs:
                                last_msg = msgs[-1]
                                if getattr(last_msg, "type", "") == "ai" and last_msg.content:
                                    yield f"data: {json.dumps({'type': 'token', 'content': last_msg.content})}\n\n"

                        yield f"data: {json.dumps(data)}\n\n"

                elif mode == "messages":
                    chunk, metadata = payload
                    node_name = metadata.get("langgraph_node") if metadata else None
                    
                    # 1. Do not stream internal/background tasks to the user
                    if node_name in ["safety", "router", "evaluator", "memory_agent"]:
                        continue
                        
                    # 2. Only stream partial chunks to avoid sending the full message again at the end,
                    # EXCEPT if the model didn't stream any chunks (non-streaming fallback).
                    is_chunk = type(chunk).__name__.endswith("Chunk")
                    if is_chunk:
                        streamed_nodes.add(node_name)
                    elif node_name in streamed_nodes:
                        # We already streamed chunks for this node, ignore the final AIMessage duplicate
                        continue
                        
                    if hasattr(chunk, "content"):
                        if chunk.content:
                            yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"
                        else:
                            import logging
                            logging.warning(f"SSE loop received empty chunk from node: {node_name}")
                            yield f"data: {json.dumps({'type': 'error', 'content': 'Warning: Received empty text chunk from model.'})}\n\n"

            # Check if execution paused
            state = orchestrator.get_state(config)
            if state.next:
                last_msg = state.values["messages"][-1]
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    yield f"data: {json.dumps({'type': 'interrupt', 'tool_calls': last_msg.tool_calls})}\n\n"



        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )

@app.post("/thread/history")
def get_thread_history(request: ThreadRequest, user_id: int = 1):
    orchestrator = get_orchestrator()
    config = {"configurable": {"thread_id": request.thread_id, "user_id": user_id}}
    
    try:
        state = orchestrator.get_state(config=config).values
        messages = state.get("messages", [])
        result = []
        for m in messages:
            if isinstance(m, HumanMessage):
                result.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                result.append({"role": "assistant", "content": m.content})
        return {"messages": result}
    except Exception as e:
        return {"messages": []}

@app.get("/threads")
def list_threads(user_id: int = 1):
    orchestrator = get_orchestrator()
    try:
        threads = list({
            c.config["configurable"]["thread_id"]
            for c in orchestrator.checkpointer.list(None)
        })
        return {"threads": threads}
    except Exception:
        return {"threads": []}

@app.delete("/thread/{thread_id}")
def delete_thread(thread_id: str, user_id: int = 1):
    """Delete thread from the database."""
    try:
        # PostgreSQL logic using the imported pool
        with pool.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))
                cursor.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (thread_id,))
                cursor.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,))
            # The pool is configured with autocommit=True, but we can explicitly commit if needed
            # conn.commit() is disabled if autocommit is true, but we'll leave it out to be safe.
            
        return {"success": True, "deleted": thread_id}
    except Exception as e:
        # Fallback to SQLite logic if pool fails (for local non-docker testing)
        try:
            orchestrator = get_orchestrator()
            conn = orchestrator.checkpointer.conn
            cursor = conn.cursor()
            cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
            cursor.execute("DELETE FROM checkpoint_blobs WHERE thread_id = ?", (thread_id,))
            cursor.execute("DELETE FROM checkpoint_writes WHERE thread_id = ?", (thread_id,))
            conn.commit()
            return {"success": True, "deleted": thread_id}
        except Exception as e2:
            print(f"DELETE ERROR: {e2}") 
            return {"success": False, "error": str(e2)}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/ingest")
def ingest_document(request: IngestRequest):
    """
    Decode the uploaded file, chunk it, embed it, and store it in the
    thread-scoped ChromaDB collection so the RAG agent can query it.
    """
    try:
        raw_bytes = base64.b64decode(request.file_bytes_b64)
        docs = process_document(raw_bytes, request.file_name)
        if not docs:
            return {"success": False, "error": "Could not parse the document. Unsupported format or empty file."}
        ok = ingest_documents(docs, request.thread_id)
        if ok:
            return {"success": True, "chunks": len(docs), "file": request.file_name}
        return {"success": False, "error": "Failed to store chunks in vector store."}
    except Exception as e:
        return {"success": False, "error": str(e)}



@app.post("/title")
def generate_title_route(request: ThreadRequest, user_id: int = 1):
    """Generate a concise 4-5 word conversation title using the LLM."""
    orchestrator = get_orchestrator()
    config = {"configurable": {"thread_id": request.thread_id}}
    try:
        state = orchestrator.get_state(config=config).values
        messages = state.get("messages", [])
        user_msgs = [
            m.content if isinstance(m.content, str) else next(
                (p["text"] for p in m.content if p.get("type") == "text"), ""
            )
            for m in messages if isinstance(m, HumanMessage)
        ]
        if not user_msgs:
            return {"title": "New Conversation"}

        from models.nvidia import get_llm
        llm = get_llm("groq/llama-3.1-8b-instant", temperature=0.3)
        combined = " | ".join(user_msgs[:6])
        prompt = (
            f"Summarize the following conversation topics into a SINGLE concise title "
            f"of exactly 4-5 words. Output ONLY the title, no punctuation, no quotes.\n\n"
            f"Topics: {combined}\n\nTitle:"
        )
        response = llm.invoke(prompt)
        title = response.content.strip().strip('"').strip("'")
        # Safety-cap at 50 chars
        return {"title": title[:50]}
    except Exception:
        return {"title": "New Conversation"}

class FeedbackRequest(BaseModel):
    thread_id: str
    question: str
    answer: str
    thumbs_up: bool
    comment: Optional[str] = ""
    eval_score: Optional[float] = 0.0

@app.post("/feedback")
def submit_feedback(request: FeedbackRequest):
    from agents.feedback_agent import log_feedback
    
    # Try to grab the latest eval score from the graph if it wasn't provided
    eval_score = request.eval_score
    if eval_score == 0.0:
        orchestrator = get_orchestrator()
        config = {"configurable": {"thread_id": request.thread_id}}
        try:
            state = orchestrator.get_state(config=config).values
            eval_score = state.get("eval_score", 0.0)
        except Exception:
            pass
            
    log_feedback(
        thread_id=request.thread_id,
        question=request.question,
        answer=request.answer,
        thumbs_up=request.thumbs_up,
        comment=request.comment,
        eval_score=eval_score
    )
    return {"success": True}

@app.get("/feedback/stats")
def get_feedback_stats_route():
    from agents.feedback_agent import get_feedback_stats
    return get_feedback_stats()
