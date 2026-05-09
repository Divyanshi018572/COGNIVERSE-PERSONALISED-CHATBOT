from langchain_core.messages import SystemMessage
from models.fallback import get_model_with_fallback
from agents.chat_agent import AgentState
from agents.tools import get_coding_tools

def coding_agent_node(state: AgentState):
    primary_llm = get_model_with_fallback("meta/llama-3.3-70b-instruct")
    fallback_llm = get_model_with_fallback("qwen/qwen2.5-coder-32b-instruct")
    
    messages = list(state["messages"])
    
    if not any(isinstance(m, SystemMessage) for m in messages):
        sys_msg = SystemMessage(content=(
            "You are an expert software engineer. You have access to a Python REPL tool, local File Management tools, and a GitHub Search tool. "
            "CRITICAL RULES: "
            "1. CODE GENERATION: When asked to write or generate code, simply output the complete code in standard markdown blocks (e.g. ```python). Do NOT invoke the execution tools automatically. "
            "2. ENGAGEMENT: After providing the code, always ask the user an engaging follow-up question (e.g., 'Should I explain this in more detail?', 'Would you like me to execute this to verify it works?', or 'Are there any specific edge cases we should handle?'). "
            "3. HUMAN-IN-THE-LOOP FOR DEBUGGING: ONLY use the Python REPL or file modification tools when you need to actively debug an issue, test a script, or if the user explicitly asks you to 'execute', 'run', or 'save' the code. "
            "4. TOOL USAGE: When you DO use tools, use the proper tool-calling API. Never output the tool call as raw JSON text in your message."
        ))
        messages = [sys_msg] + messages
        
    all_tools = get_coding_tools()
    
    # Bind tools to primary, and fall back to Qwen (without tools) if primary fails
    llm_with_tools = primary_llm.bind_tools(all_tools).with_fallbacks([fallback_llm])
    
    trace = state.get("agent_trace", []) + ["coding_agent"]
    
    response = llm_with_tools.invoke(messages)
    
    return {"messages": [response], "agent_trace": trace}
