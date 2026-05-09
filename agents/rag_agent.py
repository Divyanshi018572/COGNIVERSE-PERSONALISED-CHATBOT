from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from models.fallback import get_model_with_fallback
from core.router import TASK_MODEL_MAP
from rag.vector_store import query_documents
from agents.chat_agent import AgentState
from utils.logger import get_logger

logger = get_logger(__name__)


def rag_agent_node(state: AgentState, config: RunnableConfig):
    """
    Retrieves relevant document chunks from ChromaDB for the current thread,
    injects them as context into the system prompt, then calls the LLM.
    """
    thread_id = config.get("configurable", {}).get("thread_id", "default")

    # Get the user's latest question
    last_msg = state["messages"][-1]
    if isinstance(last_msg.content, list):
        query_text = next(
            (p["text"] for p in last_msg.content if p.get("type") == "text"), ""
        )
    else:
        query_text = last_msg.content

    # Retrieve top-k relevant chunks from the thread's Chroma collection
    context = query_documents(query_text, thread_id, k=5)

    if context:
        logger.info("rag_context_retrieved", thread_id=thread_id, chars=len(context))
        system_content = (
            "You are a precise document-analysis assistant. "
            "Answer the user's question using ONLY the document context below. "
            "If the answer is not in the context, say so clearly instead of guessing.\n\n"
            "=== DOCUMENT CONTEXT ===\n"
            f"{context}\n"
            "========================\n\n"
            "Rules:\n"
            "- Cite the source filename when referencing specific information.\n"
            "- Be concise and factual.\n"
            "- After answering, ask a follow-up question about the document to keep the conversation going."
        )
    else:
        logger.warning("rag_no_context_found", thread_id=thread_id)
        system_content = (
            "You are a helpful assistant. No document context was found for this thread. "
            "Let the user know they may need to upload a document first, or answer from general knowledge if appropriate."
        )

    messages = [SystemMessage(content=system_content)] + list(state["messages"])

    llm = get_model_with_fallback(TASK_MODEL_MAP["rag"])
    response = llm.invoke(messages)

    trace = state.get("agent_trace", []) + ["rag_agent"]
    return {
        "messages": [response],
        "rag_context": context,
        "agent_trace": trace,
    }
