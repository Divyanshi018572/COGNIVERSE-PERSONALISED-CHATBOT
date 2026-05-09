from langchain_core.messages import SystemMessage
from models.fallback import get_model_with_fallback
from agents.chat_agent import AgentState

def reasoning_agent_node(state: AgentState):
    llm = get_model_with_fallback("deepseek-ai/deepseek-r1")
    messages = state["messages"]
    
    if not any(isinstance(m, SystemMessage) for m in messages):
        sys_msg = SystemMessage(content=(
            "You are an expert analytical reasoning assistant. Think step by step. "
            "CRITICAL RULE: After providing your analysis, always ask an engaging follow-up question "
            "(e.g., asking if they need more details, alternative perspectives, or related information) to maintain continuity."
        ))
        messages = [sys_msg] + messages
        
    response = llm.invoke(messages)
    
    trace = state.get("agent_trace", []) + ["reasoning_agent"]
    return {"messages": [response], "agent_trace": trace}
