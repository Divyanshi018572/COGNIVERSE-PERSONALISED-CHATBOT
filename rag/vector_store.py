import os
import chromadb
from langchain_community.vectorstores import Chroma
from core.router import embedding_model
from utils.logger import get_logger

logger = get_logger(__name__)

# Path comes from env var — Docker volume mounts to /app/chroma_db
_chroma_path = os.getenv("CHROMA_PATH", "/app/chroma_db")
chroma_client = chromadb.PersistentClient(path=_chroma_path)

def get_vector_store(thread_id: str):
    """
    Returns a Langchain Chroma vector store specifically filtered 
    or namespaced for the given thread_id.
    We use collection names based on thread_id. Since thread_ids are UUIDs, 
    we need to ensure they are valid collection names (alphanumeric, no hyphens at start).
    """
    collection_name = f"thread_{thread_id.replace('-', '_')}"
    
    if embedding_model is None:
        logger.error("nvidia_embeddings_not_initialized_cannot_use_rag")
        return None
        
    return Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=embedding_model
    )

def ingest_documents(docs, thread_id: str) -> bool:
    """Ingests a list of Langchain Documents into the thread's vector store."""
    if not docs:
        return False
        
    vector_store = get_vector_store(thread_id)
    if vector_store is None:
        return False
        
    try:
        vector_store.add_documents(documents=docs)
        logger.info("documents_ingested_to_chroma", thread_id=thread_id, num_docs=len(docs))
        return True
    except Exception as e:
        logger.error("chroma_ingestion_failed", error=str(e), thread_id=thread_id)
        return False

def query_documents(query: str, thread_id: str, k: int = 4) -> str:
    """Queries the thread's vector store and returns concatenated text."""
    vector_store = get_vector_store(thread_id)
    if vector_store is None:
        return ""
        
    try:
        results = vector_store.similarity_search(query, k=k)
        if not results:
            return ""
            
        context = "\n\n---\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" for doc in results])
        return context
    except Exception as e:
        logger.error("chroma_query_failed", error=str(e), thread_id=thread_id)
        return ""
