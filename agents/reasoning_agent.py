from langchain_core.messages import SystemMessage
from models.fallback import get_model_with_fallback
from agents.chat_agent import AgentState

from core.router import TASK_MODEL_MAP

def reasoning_agent_node(state: AgentState):
    model_name = TASK_MODEL_MAP.get("reasoning", "groq/llama-3.3-70b-versatile")
    llm = get_model_with_fallback(model_name)
    messages = state["messages"]
    
    if not any(isinstance(m, SystemMessage) for m in messages):
        from core.prompts import FORMATTING_DIRECTIVE
        sys_msg = SystemMessage(content=(
            "You are an expert analytical reasoning assistant. Think step by step. "
            f"{FORMATTING_DIRECTIVE}\n"
            "CRITICAL RULE: After providing your analysis, always ask an engaging follow-up question "
            "(e.g., asking if they need more details, alternative perspectives, or related information) to maintain continuity."
        ))
        messages = [sys_msg] + messages
        
    # High-Resolution Self-Correction: Inject feedback if we are in a retry loop
    feedback = state.get("eval_feedback")
    if feedback and state.get("retry_count", 0) > 0:
        from langchain_core.messages import HumanMessage
        messages = list(messages)
        messages.append(HumanMessage(content=(
            f"⚠️ YOUR PREVIOUS RESPONSE FAILED QUALITY AUDIT.\n"
            f"{feedback}\n"
            "Please regenerate your response and fix ALL the issues mentioned above."
        )))

    response = llm.invoke(messages)
    
    trace = state.get("agent_trace", []) + ["reasoning_agent"]
    return {"messages": [response], "agent_trace": trace}
