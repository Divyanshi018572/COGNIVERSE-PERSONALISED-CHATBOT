from dataclasses import dataclass
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
import numpy as np
import os
from utils.logger import get_logger

logger = get_logger(__name__)

TASK_MODEL_MAP = {
    "chat":      "groq/llama-3.3-70b-versatile",
    "reasoning": "nvidia/llama-3.3-nemotron-super-49b-v1",   # DeepSeek R1 removed from NVIDIA NIM
    "coding":    "qwen/qwen2.5-coder-32b-instruct",
    "rag":       "nvidia/llama-3.3-nemotron-super-49b-v1",
    "research":  "meta/llama-3.3-70b-instruct",
    "vision":    "meta/llama-3.2-11b-vision-instruct",
    "ocr":       "meta/llama-3.2-11b-vision-instruct",
    "github":    "meta/llama-3.3-70b-instruct",
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

# ── Fast keyword pre-screens (bypass embedding model entirely) ──────────────
# These patterns reliably signal a specific agent regardless of semantic score.

REASONING_KEYWORDS = [
    "step by step", "think step", "think through", "logically",
    "deduce", "deductive", "infer", "inference", "prove that", "proof that",
    "logic puzzle", "riddle", "paradox", "thought experiment",
    "cause and effect", "why does",
    "pros and cons", "advantages and disadvantages", "compare and contrast",
    "evaluate the", "assess the", "critique", "make an argument",
    "philosophical", "moral dilemma", "ethical dilemma",
    "math proof", "derive the formula", "derivation", "calculate step",
    "all labels wrong", "labels are wrong", "wrongly labeled",
    "all signs wrong", "what must be", "what cannot be",
    "syllogism", "what follows from", "what can we conclude",
    "work through", "reason through", "break it down step",
]

CODING_KEYWORDS = [
    "write a function", "write a script", "write a program", "write a class",
    "fix this bug", "fix this code", "debug this", "refactor this",
    "implement the", "implement a",
    "sorting algorithm", "binary search", "linked list",
    "sql query", "react component", "api endpoint", "unit test",
    "mermaid diagram", "uml diagram", "architecture diagram", "class diagram",
    "flowchart", "data structure", "pseudocode", "write code",
]

RESEARCH_KEYWORDS = [
    "search the web", "latest news", "current information about", "look up",
    "browse the internet", "find online", "what happened recently", "stock price",
    "recent developments in", "who won", "search for the latest",
]

GITHUB_KEYWORDS = [
    "github.com", "github repository", "analyze this repo",
    "find top repos", "clone this repo", "pull request", "open source project",
    "contribute to this", "github topic",
]

def _keyword_route(text: str) -> str | None:
    """Fast O(n) keyword pre-screen. Returns intent name or None."""
    lower = text.lower()
    if any(kw in lower for kw in GITHUB_KEYWORDS):
        return "github"
    if any(kw in lower for kw in RESEARCH_KEYWORDS):
        return "research"
    if any(kw in lower for kw in CODING_KEYWORDS):
        return "coding"
    if any(kw in lower for kw in REASONING_KEYWORDS):
        return "reasoning"
    return None

# Define canonical intents and compute their embeddings once
INTENT_SAMPLES = {
    "coding": [
        "write a python script", "fix this bug", "debug this error",
        "how do I implement a sorting algorithm", "refactor this code",
        "write a SQL query", "create a React component", "help me with my code",
        "give me an architecture diagram", "draw a class diagram", "show data structure",
        "create a system design diagram", "explain with a diagram",
        "write unit tests", "implement a binary search tree", "optimize this function",
    ],
    "reasoning": [
        # Logic puzzles
        "think step by step through this logic puzzle",
        "deduce which box contains what based on the constraints",
        "all the labels are wrong, what does each container hold",
        "use deductive reasoning to solve this problem",
        "if all statements are false, what can we conclude",
        "work through this riddle logically",
        # Analysis & argumentation
        "analyze this argument step by step",
        "explain why this happens step by step",
        "compare and contrast these two approaches in depth",
        "evaluate the pros and cons of this decision",
        "what are the causes and effects of this situation",
        "break down the logical fallacies in this statement",
        "critique this business strategy with reasoning",
        # Math & derivation
        "derive the mathematical formula for this",
        "prove that this theorem is correct step by step",
        "solve this equation showing all your working",
        "calculate step by step why this result is correct",
        # Philosophical & ethical
        "think through this ethical dilemma",
        "what is the philosophical implication of this",
        "argue for and against this moral position",
        # Hypotheticals
        "if X then what follows logically",
        "given these constraints what must be true",
        "reason through this thought experiment",
    ],
    "research": [
        "search the web for the latest news", "find current information about",
        "what happened recently in", "look up the current stock price",
        "browse the internet for", "who won the game last night",
        "find the latest research paper on", "search online for recent developments",
    ],
    "chat": [
        "hi, how are you", "what's your name", "tell me a joke",
        "I'm feeling sad today", "hello there", "good morning",
        "what can you do", "introduce yourself",
    ],
    "github": [
        "analyze the github repository at https", "find top repositories on github for topic",
        "search github for repos", "clone this github repo", "how to contribute to this github repo",
        "github.com repository analysis", "pull request review on github",
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
    # Ensure user_input is a valid string
    if not isinstance(user_input, str):
        user_input = str(user_input or "")
    
    user_input = user_input.strip()
    if not user_input and not file_path:
        return RoutingDecision("chat", TASK_MODEL_MAP["chat"], "chat")

    if file_path:
        ext = "." + file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
        if ext in VISION_EXTS:
            return RoutingDecision("vision", TASK_MODEL_MAP["vision"], "vision")
        if ext in DOC_EXTS:
            return RoutingDecision("rag", TASK_MODEL_MAP["rag"], "rag")

    # ── Stage 1: Fast keyword pre-screen ────────────────────────────────────
    kw_intent = _keyword_route(user_input)
    if kw_intent:
        logger.info("keyword_route_success", task=kw_intent)
        return RoutingDecision(kw_intent, TASK_MODEL_MAP[kw_intent], kw_intent)

    # ── Stage 2: Semantic embedding fallback ─────────────────────────────────
    # Ensure embeddings are initialized (lazy load to avoid blocking import if API is slow)
    if embedding_model is not None:
        init_intent_embeddings()
        
    if embedding_model is not None and INTENT_EMBEDDINGS:
        try:
            # Truncate to avoid NVIDIA 512 token limit when history gets long
            truncated_input = user_input[:1500]
            input_emb = embedding_model.embed_query(truncated_input)
            
            best_intent = "chat"
            max_sim = -1.0
            
            for intent, intent_emb in INTENT_EMBEDDINGS.items():
                sim = cosine_similarity(input_emb, intent_emb)
                if sim > max_sim:
                    max_sim = sim
                    best_intent = intent
                    
            # Threshold to ensure we default to chat if nothing matches well
            if max_sim > 0.30:
                logger.info("semantic_route_success", task=best_intent, confidence=float(max_sim))
                return RoutingDecision(best_intent, TASK_MODEL_MAP[best_intent], best_intent)
        except Exception as e:
            logger.error("semantic_routing_failed", error=str(e))
    
    # Fallback to chat if API is down
    return RoutingDecision("chat", TASK_MODEL_MAP["chat"], "chat")
