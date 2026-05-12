import json
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from agents.chat_agent import AgentState
from models.fallback import get_model_with_fallback
from core.router import route, RoutingDecision, TASK_MODEL_MAP
from core.db import save_facts
from utils.logger import get_logger

logger = get_logger(__name__)

def memory_extraction_node(state: AgentState, config: RunnableConfig):
    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return {"agent_trace": state.get("agent_trace", []) + ["memory_agent"]}

    # Extract the last few messages to analyze
    messages = state.get("messages", [])
    if len(messages) < 2:
        return {"agent_trace": state.get("agent_trace", []) + ["memory_agent"]}
    
    # Get recent context (last Human and AI message)
    recent_msgs = messages[-2:]
    chat_text = "\n".join([f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in recent_msgs])

    sys_prompt = SystemMessage(content=(
        "You are a memory extraction bot. Read the following recent chat messages and extract any new, "
        "permanent facts about the user (e.g., name, preferences, location, occupation). "
        "Output ONLY a raw JSON array of strings. Do not use markdown blocks. "
        "If there are no new facts, output an empty array: []\n\n"
        "Example output: [\"User's name is Alice\", \"User prefers Python over Java\"]"
    ))
    
    user_prompt = HumanMessage(content=chat_text)
    
    model_name = TASK_MODEL_MAP.get("memory", "groq/llama-3.3-70b-versatile")
    llm = get_model_with_fallback(model_name)
    
    try:
        response = llm.invoke([sys_prompt, user_prompt])
        content = response.content.strip()
        # Clean up any potential markdown formatting
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
            
        facts = json.loads(content.strip())
        if isinstance(facts, list) and len(facts) > 0:
            logger.info("extracted_new_facts", user_id=user_id, count=len(facts))
            save_facts(user_id, facts)
    except Exception as e:
        logger.error("memory_extraction_failed", error=str(e))

    return {"agent_trace": state.get("agent_trace", []) + ["memory_agent"]}
