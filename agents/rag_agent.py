from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_community.tools.tavily_search import TavilySearchResults
from models.fallback import get_model_with_fallback
from core.router import TASK_MODEL_MAP
from rag.hybrid_retriever import hybrid_retrieve
from rag.reranker import rerank_documents
from agents.chat_agent import AgentState
from utils.logger import get_logger

logger = get_logger(__name__)


def rag_agent_node(state: AgentState, config: RunnableConfig):
    """
    Corrective RAG Agent:
    1. Hybrid Retrieve (BM25 + Vector)
    2. Rerank (NVIDIA Nemotron)
    3. If context is empty, fallback to Tavily search.
    4. Inject context and generate response.
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

    # 1. Hybrid Retrieval (Vector + BM25) -> Top 10
    logger.info("rag_hybrid_retrieval_started", thread_id=thread_id)
    retrieved_docs = hybrid_retrieve(query_text, thread_id, k=10)

    # 2. Reranking -> Top 4
    if retrieved_docs:
        logger.info("rag_reranking_started", thread_id=thread_id)
        reranked_docs = rerank_documents(query_text, retrieved_docs, top_n=4)
        context = "\n\n---\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" for doc in reranked_docs])
        logger.info("rag_context_finalized", thread_id=thread_id, num_docs=len(reranked_docs))
    else:
        context = ""

    trace = state.get("agent_trace", []) + ["rag_agent"]

    # 3. Corrective Fallback: If no local context, use Tavily
    if not context:
        logger.warning("rag_no_local_context_fallback_to_tavily", thread_id=thread_id)
        trace.append("rag_tavily_fallback")
        try:
            search = TavilySearchResults(max_results=3)
            search_results = search.invoke({"query": query_text})
            
            # Format Tavily results into context
            if isinstance(search_results, list):
                context = "\n\n---\n\n".join([f"Source (Web): {res.get('url', 'Unknown')}\n{res.get('content', '')}" for res in search_results])
            else:
                context = str(search_results)
                
            from core.prompts import FORMATTING_DIRECTIVE
            system_content = (
                "You are an expert assistant. The user's question could not be answered from local documents, "
                "so I have provided web search results below as context.\n\n"
                "=== WEB SEARCH CONTEXT ===\n"
                f"{context}\n"
                "==========================\n\n"
                f"{FORMATTING_DIRECTIVE}\n\n"
                "Rules:\n"
                "- Cite the web source URL when referencing specific information.\n"
                "- Be concise and factual.\n"
                "- After answering, ask an engaging follow-up question."
            )
        except Exception as e:
            logger.error("rag_tavily_fallback_failed", error=str(e))
            system_content = (
                "You are a helpful assistant. No document context or web context could be found for this query. "
                "Answer from general knowledge if appropriate, and let the user know they may need to provide more info."
            )
    else:
        # Local context was found
        from core.prompts import FORMATTING_DIRECTIVE
        system_content = (
            "You are a strict document synthesis engine. Your only function is to "
            "extract and combine factual information from the provided context chunks.\n\n"
            "## ABSOLUTE OUTPUT RULES\n"
            "1. ONLY use information explicitly stated in the provided context\n"
            "2. NEVER ask follow-up questions — not at the end, not anywhere\n"
            "3. NEVER speculate about future trends, next steps, or what could happen\n"
            "4. NEVER add phrases like: 'What do you think about...', 'Would you like to know more...', 'In the next 5 years...'\n"
            "5. NEVER add a conclusion paragraph that goes beyond the source material\n"
            "6. NEVER add commentary, opinions, or suggestions\n"
            "7. If the context does not contain enough information to answer, respond ONLY with: "
            "'The provided context does not contain sufficient information to answer this question.'\n\n"
            "## OUTPUT FORMAT\n"
            "- Answer in plain, dry, declarative sentences only\n"
            "- Each claim must be directly traceable to the provided context\n"
            "- Stop writing the moment the factual synthesis is complete\n"
            "- No closing remarks, no summaries beyond what context states, no transitional phrases\n\n"
            f"{FORMATTING_DIRECTIVE}\n\n"
            "## CONTEXT WINDOW\n"
            f"{context}\n\n"
            "## YOUR RESPONSE\n"
            "(Begin directly. No preamble. Stop at the last factual point.)"
        )

    messages = [SystemMessage(content=system_content)] + list(state["messages"])

    # High-Resolution Self-Correction: Inject feedback if we are in a retry loop
    feedback = state.get("eval_feedback")
    if feedback and state.get("retry_count", 0) > 0:
        messages.append(HumanMessage(content=(
            f"⚠️ YOUR PREVIOUS RESPONSE FAILED QUALITY AUDIT.\n"
            f"{feedback}\n"
            "Please regenerate your response and fix ALL the issues mentioned above."
        )))

    llm = get_model_with_fallback(TASK_MODEL_MAP["rag"])
    response = llm.invoke(messages, config)

    if not response.content or not response.content.strip():
        raise ValueError("RAG agent generated an empty or whitespace-only response from NVIDIA NIM.")

    return {
        "messages": [response],
        "rag_context": context,
        "agent_trace": trace,
    }
