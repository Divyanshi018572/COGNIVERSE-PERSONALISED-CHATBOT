from langchain_nvidia_ai_endpoints import ChatNVIDIA
import os
from dotenv import load_dotenv

load_dotenv()

try:
    llm = ChatNVIDIA(
        model="nvidia/llama-3.3-nemotron-super-49b-v1",
        api_key=os.getenv("NVIDIA_API_KEY"),
        temperature=0.7
    )
    res = llm.invoke("Hello, who are you?")
    print("Content:", repr(res.content))
except Exception as e:
    print("Error:", type(e).__name__, e)
