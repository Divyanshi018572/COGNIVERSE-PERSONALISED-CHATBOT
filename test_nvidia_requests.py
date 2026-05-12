import requests
import os
from dotenv import load_dotenv

load_dotenv()

headers = {
    "Authorization": f"Bearer {os.getenv('NVIDIA_API_KEY')}",
    "Content-Type": "application/json"
}
data = {
    "model": "meta/llama-3.3-70b-instruct",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 10
}
res = requests.post("https://integrate.api.nvidia.com/v1/chat/completions", headers=headers, json=data)
print("meta/llama-3.3-70b-instruct:", res.status_code, res.text)

data["model"] = "nvidia/llama-3.3-nemotron-super-49b-v1"
res = requests.post("https://integrate.api.nvidia.com/v1/chat/completions", headers=headers, json=data)
print("nvidia/llama-3.3-nemotron-super-49b-v1:", res.status_code, res.text)
