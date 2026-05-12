from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from utils.logger import get_logger
from rag.vector_store import get_vector_store
import math

logger = get_logger(__name__)

def reciprocal_rank_fusion(results: list[list[Document]], k=60) -> list[Document]:
    """
    Reciprocal Rank Fusion (RRF) algorithm to combine multiple ranked lists.
    results: List of ranked document lists.
    k: RRF constant.
    """
    fused_scores = {}
    doc_map = {}
    
    for ranked_list in results:
        for rank, doc in enumerate(ranked_list):
            doc_id = doc.metadata.get("source", "") + str(doc.page_content[:50])
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0.0
            
            fused_scores[doc_id] += 1.0 / (rank + k)

    # Sort by fused score
    reranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[doc_id] for doc_id, _ in reranked]

def hybrid_retrieve(query: str, thread_id: str, k: int = 5) -> list[Document]:
    """
    Retrieves documents using both Vector Search (Chroma) and Keyword Search (BM25),
    then fuses the results using RRF.
    """
    vector_store = get_vector_store(thread_id)
    if not vector_store:
        logger.warning("hybrid_retrieval_failed_no_vector_store", thread_id=thread_id)
        return []
        
    try:
        # 1. Vector Search
        # We retrieve slightly more for fusion
        vector_results = vector_store.similarity_search(query, k=k*2)
        
        # 2. BM25 Search (Keyword)
        # We need all documents in the thread to build the BM25 index on the fly.
        # Note: In a large system, BM25 indices should be persisted. For this agentic
        # context scoped to a thread, fetching all thread docs is feasible.
        all_docs_dict = vector_store.get()
        all_docs = []
        if all_docs_dict and "documents" in all_docs_dict:
            for idx, content in enumerate(all_docs_dict["documents"]):
                metadata = all_docs_dict["metadatas"][idx] if "metadatas" in all_docs_dict else {}
                all_docs.append(Document(page_content=content, metadata=metadata))
                
        if not all_docs:
            return vector_results[:k]
            
        tokenized_corpus = [doc.page_content.lower().split(" ") for doc in all_docs]
        bm25 = BM25Okapi(tokenized_corpus)
        
        tokenized_query = query.lower().split(" ")
        bm25_scores = bm25.get_scores(tokenized_query)
        
        # Get top k indices from BM25
        top_n = min(k*2, len(all_docs))
        bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_n]
        bm25_results = [all_docs[i] for i in bm25_indices]
        
        # 3. Fuse Results
        fused_docs = reciprocal_rank_fusion([vector_results, bm25_results], k=60)
        
        logger.info("hybrid_retrieval_completed", thread_id=thread_id, query=query)
        return fused_docs[:k]
        
    except Exception as e:
        logger.error("hybrid_retrieval_error", error=str(e), thread_id=thread_id)
        # Fallback to standard vector search
        return vector_store.similarity_search(query, k=k)
