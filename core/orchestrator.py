from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage

from agents.chat_agent import AgentState, chat_agent_node
from agents.reasoning_agent import reasoning_agent_node
from agents.coding_agent import coding_agent_node
from agents.research_agent import research_agent_node
from agents.memory_agent import memory_extraction_node
from agents.safety_agent import check_safety
from agents.tool_node import coding_tool_node
from agents.rag_agent import rag_agent_node
from core.router import route

# ── 1. Graph State Setup ──────────────────────────────────────────────────────
graph = StateGraph(AgentState)

# ── 2. Node Definitions ───────────────────────────────────────────────────────
def safety_node(state: AgentState):
    user_msg_content = state["messages"][-1].content
    if isinstance(user_msg_content, list):
        user_msg = next((item["text"] for item in user_msg_content if item.get("type") == "text"), "")
    else:
        user_msg = user_msg_content
        
    is_safe, reason = check_safety(user_msg)
    return {"blocked": not is_safe, "block_reason": reason if not is_safe else ""}

def blocked_node(state: AgentState):
    msg = AIMessage(content=f"⚠️ Message Blocked: {state['block_reason']}")
    return {"messages": [msg], "agent_trace": state.get("agent_trace", []) + ["blocked_node"]}

def router_node(state: AgentState):
    user_msg = state["messages"][-1].content
    if isinstance(user_msg, list):
        return {"task": "vision"}
    decision = route(user_msg, state.get("file_path"))
    return {"task": decision.task}

# Add nodes to graph
graph.add_node("safety", safety_node)
graph.add_node("blocked", blocked_node)
graph.add_node("router", router_node)
graph.add_node("chat_agent", chat_agent_node)
graph.add_node("reasoning_agent", reasoning_agent_node)
graph.add_node("coding_agent", coding_agent_node)
graph.add_node("coding_tools", coding_tool_node)
graph.add_node("research_agent", research_agent_node)
graph.add_node("rag_agent", rag_agent_node)
graph.add_node("memory_agent", memory_extraction_node)
from agents.vision_agent import vision_agent_node
graph.add_node("vision_agent", vision_agent_node)

# ── 3. Edges & Routing ────────────────────────────────────────────────────────
def safety_route(state: AgentState):
    if state.get("blocked", False):
        return "blocked"
    return "router"

def task_route(state: AgentState):
    task = state.get("task", "chat")
    if task == "coding":
        return "coding_agent"
    elif task == "reasoning":
        return "reasoning_agent"
    elif task == "research":
        return "research_agent"
    elif task == "vision":
        return "vision_agent"
    elif task == "rag":
        return "rag_agent"
    return "chat_agent"

# Edges
graph.add_edge(START, "safety")
graph.add_conditional_edges("safety", safety_route)
graph.add_edge("blocked", END)

graph.add_conditional_edges("router", task_route)

# Tool execution routing
def tools_condition(state: AgentState):
    messages = state.get("messages", [])
    if not messages:
        return END
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "coding_tools"
    return END

graph.add_conditional_edges("coding_agent", tools_condition)
graph.add_edge("coding_tools", "coding_agent")

# All agents route to memory_agent to extract facts, then to END
graph.add_edge("chat_agent", "memory_agent")
graph.add_edge("reasoning_agent", "memory_agent")
graph.add_edge("research_agent", "memory_agent")
graph.add_edge("rag_agent", "memory_agent")
graph.add_edge("vision_agent", "memory_agent")
graph.add_edge("memory_agent", END)

from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
import os

# ── 4. Compile Graph ──────────────────────────────────────────────────────────
POSTGRES_URI = os.getenv("POSTGRES_URI", "postgresql://user:password@localhost:5432/chatbot_db")

pool = ConnectionPool(
    conninfo=POSTGRES_URI,
    max_size=20,
    kwargs={"autocommit": True}
)

def get_orchestrator():
    # In a real app we'd want to manage pool lifecycle, 
    # but for simplicity we'll create the checkpointer per request or keep it global
    checkpointer = PostgresSaver(pool)
    checkpointer.setup() # create tables if they don't exist
    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["coding_tools"]
    )
