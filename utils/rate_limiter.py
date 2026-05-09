import time
import os
import redis
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Model rate limits (NVIDIA NIM free tier: 40 RPM) ─────────────────────────
RATE_LIMITS = {
    "meta/llama-3.3-70b-instruct":            35,
    "deepseek-ai/deepseek-r1":                35,
    "qwen/qwen2.5-coder-32b-instruct":        35,
    "nvidia/llama-3.3-nemotron-super-49b-v1": 35,
    "microsoft/phi-4":                        35,
    "nvidia/nv-embedqa-e5-v5":                35,
    "nvidia/llama-nemotron-rerank-1b-v2":     35,
    "meta/llama-3.2-90b-vision-instruct":     35,
    "meta/llama-3.2-11b-vision-instruct":     35,
    # Groq fallbacks
    "groq/llama-3.3-70b-versatile":           4,
    # Google fallbacks
    "gemini-1.5-flash":                       14,
}

# ── Primary → Fallback chain ──────────────────────────────────────────────────
FALLBACK_CHAIN = {
    "meta/llama-3.3-70b-instruct":            "groq/llama-3.3-70b-versatile",
    "deepseek-ai/deepseek-r1":                "nvidia/llama-3.3-nemotron-super-49b-v1",
    "qwen/qwen2.5-coder-32b-instruct":        "microsoft/phi-4",
    "nvidia/llama-3.3-nemotron-super-49b-v1": "meta/llama-3.3-70b-instruct",
    "meta/llama-3.2-90b-vision-instruct":     "meta/llama-3.2-11b-vision-instruct",
    "groq/llama-3.3-70b-versatile":           "gemini-1.5-flash",
    "gemini-1.5-flash":                       "microsoft/phi-4",
}

class RateLimiter:
    """
    Redis-backed sliding-window rate limiter.
    Tracks request timestamps per model over the last 60 seconds.
    """
    def __init__(self):
        redis_uri = os.getenv("REDIS_URI", "redis://localhost:6379/0")
        try:
            self.r = redis.from_url(redis_uri, decode_responses=True)
            self.r.ping() # test connection
        except Exception as e:
            logger.error("redis_connection_failed", error=str(e))
            self.r = None

    def is_available(self, model: str) -> bool:
        if self.r is None:
            # Fallback to true if Redis is down, we'll hit the API and let Tenacity handle 429s
            return True
            
        limit = RATE_LIMITS.get(model, 35)
        now = time.time()
        window_start = now - 60
        
        key = f"rate_limit:{model}"
        
        # Clean up old records
        try:
            self.r.zremrangebyscore(key, 0, window_start)
            current_count = self.r.zcard(key)
            return current_count < limit
        except Exception:
            return True

    def record_request(self, model: str) -> None:
        if self.r is None:
            return
            
        now = time.time()
        key = f"rate_limit:{model}"
        
        try:
            # Add current timestamp
            # Score and value are both the timestamp
            self.r.zadd(key, {str(now): now})
            # Expire the key after 60 seconds to save memory
            self.r.expire(key, 60)
        except Exception as e:
            logger.error("redis_record_failed", error=str(e))

    def get_available_model(self, primary: str) -> str:
        model = primary
        visited: set[str] = set()
        while model and model not in visited:
            if self.is_available(model):
                if model != primary:
                    logger.info("rate_limit_fallback", primary=primary, using=model)
                return model
            visited.add(model)
            model = FALLBACK_CHAIN.get(model, "")
        logger.warning("all_fallbacks_exhausted", primary=primary)
        return "microsoft/phi-4"

rate_limiter = RateLimiter()
