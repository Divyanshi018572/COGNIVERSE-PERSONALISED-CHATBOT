from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph.message import add_messages
from models.fallback import get_model_with_fallback
from agents.chat_agent import AgentState

def vision_agent_node(state: AgentState):
    llm = get_model_with_fallback("meta/llama-3.2-11b-vision-instruct")
    messages = state["messages"]
    
    # Prepend a system message if one doesn't exist
    if not any(isinstance(m, SystemMessage) for m in messages):
        sys_msg = SystemMessage(content=(
            "You are a helpful AI assistant with advanced computer vision capabilities. Describe the provided image accurately and answer the user's questions about it. "
            "CRITICAL RULE: After analyzing the image, always ask an engaging follow-up question "
            "(e.g., asking if they want specific details analyzed, related information, or if they have other images) to maintain chat continuity."
        ))
        messages = [sys_msg] + messages
        
    response = llm.invoke(messages)
    
    trace = state.get("agent_trace", []) + ["vision_agent"]
    return {"messages": [response], "agent_trace": trace}
