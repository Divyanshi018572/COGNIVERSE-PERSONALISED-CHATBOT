import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError
from models.nvidia import get_llm
from utils.rate_limiter import rate_limiter
from utils.logger import get_logger

logger = get_logger(__name__)


def get_model_with_fallback(primary: str, temperature: float = 0.7):
    model = rate_limiter.get_available_model(primary)
    rate_limiter.record_request(model)
    return get_llm(model, temperature)


@retry(
    retry=retry_if_exception_type(RateLimitError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def invoke_with_retry(llm, messages):
    return llm.invoke(messages)
