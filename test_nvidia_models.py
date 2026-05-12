from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="meta/llama-3.1-70b-instruct",
    openai_api_base="https://integrate.api.nvidia.com/v1",
    openai_api_key=os.getenv("NVIDIA_API_KEY"),
    max_retries=1
)
print("Invoke 1...")
try:
    res = llm.invoke("Hello")
    print(repr(res.content))
except Exception as e:
    print("Error 1:", type(e).__name__, e)

llm2 = ChatOpenAI(
    model="nvidia/llama-3.1-nemotron-70b-instruct",
    openai_api_base="https://integrate.api.nvidia.com/v1",
    openai_api_key=os.getenv("NVIDIA_API_KEY"),
    max_retries=1
)
print("Invoke 2...")
try:
    res2 = llm2.invoke("Hello")
    print(repr(res2.content))
except Exception as e:
    print("Error 2:", type(e).__name__, e)
