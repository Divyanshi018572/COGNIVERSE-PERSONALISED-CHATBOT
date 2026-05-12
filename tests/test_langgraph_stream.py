from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage

class State(TypedDict):
    messages: Annotated[list, add_messages]

def node_a(state):
    return {"messages": [AIMessage(content="Hello from Node A")]}

graph = StateGraph(State)
graph.add_node("node_a", node_a)
graph.add_edge(START, "node_a")
graph.add_edge("node_a", END)
app = graph.compile()

for mode, payload in app.stream({"messages": [HumanMessage(content="Hi")]}, stream_mode=["messages", "updates"]):
    if mode == "messages":
        chunk, metadata = payload
        print(f"MESSAGES mode -> Type: {type(chunk).__name__}, content: {chunk.content}, metadata: {metadata}")
    elif mode == "updates":
        print(f"UPDATES mode -> {payload}")
