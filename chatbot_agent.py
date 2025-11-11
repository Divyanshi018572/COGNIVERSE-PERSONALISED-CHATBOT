
import os
from langgraph.graph import StateGraph, START , END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3





load_dotenv()

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]

llm = ChatOpenAI(
    model="meta-llama/llama-3.3-70b-instruct:free",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.7)

#add nodes
def chat_node(state:ChatState):

    messages = state["messages"]
    # send to llm
    response = llm.invoke(messages)
    return {"messages":[response]}

conn = sqlite3.connect(database = "chatbot.db",check_same_thread = False)
checkpointer = SqliteSaver(conn = conn)
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_edge(START,"chat_node")
graph.add_edge("chat_node",END)

chatbot = graph.compile(checkpointer=checkpointer)

#state ={
#"messages": [HumanMessage(content="what is my name")]}
#config = {"configurable":{"thread_id":"thread_3"}}


#response = chatbot.invoke(state, config = config)["messages"][-1].content
#print(response)
def store_info():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


"""while True:
    user_message = input("type your message: ")
    print("user:", user_message)
    if user_message.strip().lower() == "exit":
        break
    config1 = {"configurable":{"thread_id":"thread_1"}}"""

    #state["messages"].append(HumanMessage(content=user_message))

    #print("AI:", state["messages"][-1].content)










