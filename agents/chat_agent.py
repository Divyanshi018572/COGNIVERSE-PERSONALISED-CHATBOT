from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph.message import add_messages
from models.fallback import get_model_with_fallback
from core.router import route, RoutingDecision, TASK_MODEL_MAP

class AgentState(TypedDict):
    messages:       Annotated[list[BaseMessage], add_messages]
    task:           str           
    file_path:      str | None    
    file_bytes:     bytes | None  
    blocked:        bool          
    block_reason:   str           
    hitl_needed:    bool          
    hitl_question:  str           
    hitl_response:  str           
    eval_score:     float         
    eval_feedback:  str           
    rag_context:    str           
    agent_trace:    list[str]     


from langchain_core.runnables import RunnableConfig
from core.db import get_facts

def chat_agent_node(state: AgentState, config: RunnableConfig):
    llm = get_model_with_fallback(TASK_MODEL_MAP["chat"])
    messages = state["messages"]
    
    # Prepend a system message if one doesn't exist
    if not any(isinstance(m, SystemMessage) for m in messages):
        # Fetch user facts from LTM
        user_id = config.get("configurable", {}).get("user_id")
        memory_context = ""
        if user_id:
            facts = get_facts(user_id)
            if facts:
                memory_context = f"\n\nHere is what you know about the user:\n{facts}\n"

        sys_msg = SystemMessage(content=(
            "You are a helpful conversational AI assistant. "
            f"{memory_context}"
            "CRITICAL RULE: Always end your response with an engaging follow-up question related to the user's query "
            "to maintain continuity and keep the user engaged."
        ))
        messages = [sys_msg] + messages
        
    response = llm.invoke(messages)
    
    trace = state.get("agent_trace", []) + ["chat_agent"]
    return {"messages": [response], "agent_trace": trace}
