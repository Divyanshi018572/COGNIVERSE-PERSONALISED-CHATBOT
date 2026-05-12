from langchain_nvidia_ai_endpoints import NVIDIARerank
import os
from utils.logger import get_logger

logger = get_logger(__name__)

_reranker_instance = None

def get_reranker() -> NVIDIARerank:
    global _reranker_instance
    if _reranker_instance is None:
        try:
            _reranker_instance = NVIDIARerank(
                model="nvidia/llama-nemotron-rerank-1b-v2",
                # The NVIDIA API key should be in the environment
            )
            logger.info("nvidia_reranker_initialized")
        except Exception as e:
            logger.error("failed_to_initialize_reranker", error=str(e))
            return None
    return _reranker_instance

def rerank_documents(query: str, docs: list, top_n: int = 4) -> list:
    """
    Reranks a list of Langchain Documents based on relevance to the query.
    Uses NVIDIA Nemotron 1B reranker.
    """
    if not docs:
        return []
        
    reranker = get_reranker()
    if not reranker:
        logger.warning("reranker_unavailable_returning_original_docs")
        return docs[:top_n]
        
    try:
        # NVIDIARerank provides a compress_documents method compatible with DocumentCompressor
        reranked_docs = reranker.compress_documents(
            documents=docs,
            query=query
        )
        logger.info("documents_reranked", original_count=len(docs), reranked_count=len(reranked_docs))
        return reranked_docs[:top_n]
    except Exception as e:
        logger.error("reranking_failed", error=str(e))
        return docs[:top_n]
