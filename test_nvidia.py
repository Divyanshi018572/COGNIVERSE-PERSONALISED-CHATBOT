from models.nvidia import get_llm
from langchain_core.messages import HumanMessage

llm = get_llm("nvidia/llama-3.3-nemotron-super-49b-v1")
res = llm.invoke([HumanMessage(content="Hello")])
print(repr(res.content))
