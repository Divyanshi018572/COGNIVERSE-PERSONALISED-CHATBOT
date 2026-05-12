import os
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, NVIDIARerank
from dotenv import load_dotenv

load_dotenv()

NVIDIA_BASE = "https://integrate.api.nvidia.com/v1"
NVIDIA_KEY  = os.getenv("NVIDIA_API_KEY")
GROQ_KEY    = os.getenv("GROQ_API_KEY")
GOOGLE_KEY  = os.getenv("GOOGLE_API_KEY")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")

def get_llm(model: str, temperature: float = 0.7) -> ChatOpenAI:
    """Route to correct provider based on model prefix."""
    if model.startswith("groq/"):
        return ChatGroq(
            model=model.replace("groq/", ""),
            groq_api_key=GROQ_KEY,
            temperature=temperature,
        )
    if model.startswith("gemini"):
        # Ensure we use a valid model name for v1beta API
        if model == "gemini-1.5-flash":
            model = "gemini-1.5-flash-latest"
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=GOOGLE_KEY,
            temperature=temperature,
        )
    if "llama" in model.lower() and "instruct:free" in model.lower():
        # Fallback to OpenRouter for the free llama model
        openrouter_model = model.replace("meta/", "meta-llama/")
        return ChatOpenAI(
            model=openrouter_model,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=OPENROUTER_KEY,
            temperature=temperature,
            max_retries=0,
            request_timeout=60,
        )
        
    if "qwen" in model.lower():
        return ChatOpenAI(
            model=model,
            openai_api_base=NVIDIA_BASE,
            openai_api_key=NVIDIA_KEY,
            temperature=temperature,
            max_retries=0,
            request_timeout=120,
            max_tokens=4096,
            streaming=False,
            model_kwargs={"stream": False}
        )

    # Default: NVIDIA NIM
    return ChatOpenAI(
        model=model,
        openai_api_base=NVIDIA_BASE,
        openai_api_key=NVIDIA_KEY,
        temperature=temperature,
        max_retries=0,
        request_timeout=120,
        max_tokens=2048,
        streaming=False,
        model_kwargs={"stream": False}
    )


def get_embeddings() -> NVIDIAEmbeddings:
    return NVIDIAEmbeddings(
        model="nvidia/nv-embedqa-e5-v5",
        api_key=NVIDIA_KEY,
        truncate="END",
    )


def get_reranker() -> NVIDIARerank:
    return NVIDIARerank(
        model="nvidia/llama-nemotron-rerank-1b-v2",
        api_key=NVIDIA_KEY,
        top_n=5,
    )
