
import os
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

load_dotenv()  # Load API key

# Chat state to store conversation history
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Initialize LLM (Meta LLaMA via OpenRouter)
llm = ChatOpenAI(
    model="meta-llama/llama-3.3-70b-instruct:free",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.7
)

# Node: send messages to LLM and return response
def chat_node(state: ChatState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# SQLite checkpoint setup
conn = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# Define and build conversation graph
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
chatbot = graph.compile(checkpointer=checkpointer)

# List stored chat threads from DB
def store_info():
    return list({
        c.config["configurable"]["thread_id"]
        for c in checkpointer.list(None)
    })










