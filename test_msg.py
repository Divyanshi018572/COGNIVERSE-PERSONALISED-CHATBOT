from langchain_core.messages import AIMessage
m = AIMessage(content="hello")
print(getattr(m, "type", ""))
print(type(m))
