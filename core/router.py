from dataclasses import dataclass
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
import numpy as np
import os
from utils.logger import get_logger

logger = get_logger(__name__)

TASK_MODEL_MAP = {
    "chat":      "groq/llama-3.3-70b-versatile",  # Switched to latest Groq model
    "reasoning": "deepseek-ai/deepseek-r1",
    "coding":    "qwen/qwen2.5-coder-32b-instruct",
    "rag":       "nvidia/llama-3.3-nemotron-super-49b-v1",
    "research":  "meta/llama-3.3-70b-instruct",
    "vision":    "meta/llama-3.2-90b-vision-instruct",
    "ocr":       "meta/llama-3.2-90b-vision-instruct",
}

VISION_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
DOC_EXTS    = {".pdf", ".docx", ".txt", ".csv", ".xlsx"}

# Initialize NVIDIA API-based embedding model
# This uses the free NIM tier and removes the need for local PyTorch dependencies
try:
    embedding_model = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")
except Exception as e:
    logger.error("failed_to_initialize_nvidia_embeddings", error=str(e))
    embedding_model = None

# Define canonical intents and compute their embeddings once
INTENT_SAMPLES = {
    "coding": [
        "write a python script", "fix this bug", "debug this error",
        "how do I implement a sorting algorithm", "refactor this code",
        "write a SQL query", "create a React component", "help me with my code"
    ],
    "reasoning": [
        "analyze this argument", "explain why this happens step by step",
        "compare and contrast these two approaches", "evaluate the pros and cons",
        "think through this logical puzzle", "derive the mathematical formula",
        "what are the causes of"
    ],
    "research": [
        "search the web for the latest news", "find current information about",
        "what happened recently in", "look up the current stock price",
        "browse the internet for", "who won the game last night"
    ],
    "chat": [
        "hi, how are you", "what's your name", "tell me a joke",
        "I'm feeling sad today", "hello there", "good morning"
    ]
}

# Pre-compute intent embeddings
INTENT_EMBEDDINGS = {}

def init_intent_embeddings():
    global INTENT_EMBEDDINGS
    if embedding_model is not None and not INTENT_EMBEDDINGS:
        try:
            for intent, samples in INTENT_SAMPLES.items():
                embeddings = embedding_model.embed_documents(samples)
                INTENT_EMBEDDINGS[intent] = np.mean(embeddings, axis=0)
            logger.info("intent_embeddings_initialized_successfully")
        except Exception as e:
            logger.error("failed_to_precompute_intent_embeddings", error=str(e))

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@dataclass
class RoutingDecision:
    task: str
    model: str
    agent: str

def route(user_input: str, file_path: str | None = None) -> RoutingDecision:
    if file_path:
        ext = "." + file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
        if ext in VISION_EXTS:
            return RoutingDecision("vision", TASK_MODEL_MAP["vision"], "vision")
        if ext in DOC_EXTS:
            return RoutingDecision("rag", TASK_MODEL_MAP["rag"], "rag")

    # Ensure embeddings are initialized (lazy load to avoid blocking import if API is slow)
    if embedding_model is not None:
        init_intent_embeddings()
        
    if embedding_model is not None and INTENT_EMBEDDINGS:
        try:
            input_emb = embedding_model.embed_query(user_input)
            
            best_intent = "chat"
            max_sim = -1.0
            
            for intent, intent_emb in INTENT_EMBEDDINGS.items():
                sim = cosine_similarity(input_emb, intent_emb)
                if sim > max_sim:
                    max_sim = sim
                    best_intent = intent
                    
            # Threshold to ensure we default to chat if nothing matches well
            if max_sim > 0.35:
                logger.info("semantic_route_success", task=best_intent, confidence=float(max_sim))
                return RoutingDecision(best_intent, TASK_MODEL_MAP[best_intent], best_intent)
        except Exception as e:
            logger.error("semantic_routing_failed", error=str(e))
    
    # Fallback to chat if API is down
    return RoutingDecision("chat", TASK_MODEL_MAP["chat"], "chat")
