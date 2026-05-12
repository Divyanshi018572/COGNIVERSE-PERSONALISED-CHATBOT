from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from models.fallback import get_model_with_fallback
from agents.chat_agent import AgentState
import os
import json

from core.router import TASK_MODEL_MAP

def research_agent_node(state: AgentState):
    model_name = TASK_MODEL_MAP.get("research", "groq/llama-3.3-70b-versatile")
    llm = get_model_with_fallback(model_name)
    messages = list(state["messages"])
    
    if not any(isinstance(m, SystemMessage) for m in messages):
        from core.prompts import FORMATTING_DIRECTIVE
        sys_msg = SystemMessage(content=(
            "You are an expert research assistant. Use the ArXiv tool for academic/scientific papers, and Tavily for general web searches. "
            "If the user asks a follow-up question about previous papers (like asking for links), DO NOT search again if you already have the information. "
            "Always provide ArXiv links in the format https://arxiv.org/abs/<id>. "
            f"{FORMATTING_DIRECTIVE}\n"
            "CRITICAL RULE: Always end your response with an engaging follow-up question "
            "(e.g., asking if they want a deeper summary, related research, or clarification on specific findings) to keep the user engaged."
        ))
        messages = [sys_msg] + messages
    
    search = TavilySearchResults(max_results=3)
    arxiv_wrapper = ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=2000)
    arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
    
    tools = [search, arxiv]
    llm_with_tools = llm.bind_tools(tools)
    
    trace = state.get("agent_trace", []) + ["research_agent"]
    
    # First LLM call
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

    response = llm_with_tools.invoke(messages)
    
    if not response.tool_calls:
        return {"messages": [response], "agent_trace": trace}
        
    # If the LLM decided to call tools, execute them
    new_messages = [response]
    messages.append(response)
    
    for tool_call in response.tool_calls:
        try:
            if tool_call["name"] == "tavily_search_results_json":
                tool_res = search.invoke(tool_call["args"])
            elif tool_call["name"] == "arxiv":
                tool_res = arxiv.invoke(tool_call["args"])
            else:
                tool_res = "Unknown tool."
        except Exception as e:
            tool_res = f"Tool error: {str(e)}"
            
        tool_msg = ToolMessage(content=str(tool_res), tool_call_id=tool_call["id"])
        new_messages.append(tool_msg)
        messages.append(tool_msg)
        
    # Second LLM call with tool results
    final_response = llm_with_tools.invoke(messages)
    new_messages.append(final_response)
    
    return {"messages": new_messages, "agent_trace": trace}
