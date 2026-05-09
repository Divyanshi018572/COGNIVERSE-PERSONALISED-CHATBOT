from langchain_core.messages import HumanMessage
from models.fallback import get_model_with_fallback, invoke_with_retry
from utils.logger import get_logger

logger = get_logger(__name__)

# Fast keyword blocklist — no LLM call needed for obvious violations
BLOCKED_PATTERNS = [
    "how to make a bomb", "how to hack", "child abuse",
    "create malware", "ddos attack", "generate fake id",
]

SAFETY_PROMPT = """You are a content safety classifier. Respond with exactly:
SAFE — if the message is appropriate
UNSAFE: <brief reason> — if it violates guidelines

Message: {message}
Classification:"""


def check_safety(user_input: str) -> tuple[bool, str]:
    """Returns (is_safe, reason). Fast path first, LLM second."""
    lower = user_input.lower()
    for pattern in BLOCKED_PATTERNS:
        if pattern in lower:
            return False, f"Blocked: {pattern}"
    try:
        # Use an ultra-fast model for safety checks to avoid lagging the conversation
        llm = get_model_with_fallback("groq/llama-3.1-8b-instant", temperature=0.0)
        prompt = SAFETY_PROMPT.format(message=user_input[:500])
        response = invoke_with_retry(llm, prompt)
        result = response.content.strip()
        if result.startswith("UNSAFE"):
            reason = result.replace("UNSAFE:", "").strip()
            logger.warning("safety_blocked", reason=reason)
            return False, reason
        return True, "safe"
    except Exception as e:
        logger.error("safety_check_error", error=str(e))
        # Fail open instead of failing closed, so the app remains usable even if the safety LLM times out
        return True, "safe"
