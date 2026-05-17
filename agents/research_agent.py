from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from models.fallback import get_model_with_fallback
from agents.chat_agent import AgentState
import os
import json

from core.router import TASK_MODEL_MAP

# ── Research Agent System Prompt ─────────────────────────────────────────────
RESEARCH_SYSTEM_PROMPT = """You are COGNIVERSE Research Agent — an elite AI journalist and analyst.
You have access to real-time web search (Tavily) and academic paper search (ArXiv).

## YOUR RESPONSE STRUCTURE (ALWAYS follow this):

### 📰 [Topic Heading]
Start with a **1-sentence TL;DR** summary in bold.

### 🔍 Key Findings
Use bullet points with **bold labels**. Each bullet = one distinct finding.
Example:
- **Market Cap**: Bitcoin hit $X trillion as of [date]
- **Price Movement**: BTC up X% in the last 24 hours

### 📊 Data Table (when numbers/comparisons exist)
Use a markdown table to present structured data cleanly.
| Metric | Value | Change |
|--------|-------|--------|
| ...    | ...   | ...    |

### 🌐 Sources
List ONLY the title and URL — NO raw JSON, NO Python dict syntax:
- [Source Title](URL)
- [Source Title](URL)

### 💡 Key Takeaway
End with a 1-2 sentence insight or implication.

---
## STRICT RULES:
1. NEVER output raw Python dicts like `{'title': ..., 'url': ..., 'content': ...}`
2. NEVER output `\\n` as literal text — use actual newlines
3. NEVER repeat the raw search result verbatim — always synthesize and summarize
4. NEVER output the tool call result as-is — transform it into readable prose/bullets
5. Format ALL numbers with commas (e.g., $78,047 not $78047)
6. Always cite your sources using clickable markdown links [Title](URL)
7. If data is real-time (prices, scores), mention the timestamp from the source
8. Use bold **text** for all important terms, numbers, and entities
"""

RESEARCH_SYNTHESIS_PROMPT = """Now synthesize the search results above into a clean, well-formatted report.

CRITICAL: 
- Do NOT include any raw Python dict output like {{'title':...,'url':...}}
- Do NOT output literal \\n characters  
- Format sources as clickable markdown: [Title](URL)
- Use bullet points, bold text, and markdown tables
- Provide a clear summary with key insights
- End with ONE engaging follow-up question to keep the conversation going
"""


def _format_tavily_results(raw_results) -> str:
    """Converts raw Tavily list-of-dicts into clean readable text for the LLM."""
    if isinstance(raw_results, str):
        # Already stringified — try to parse it
        try:
            raw_results = json.loads(raw_results.replace("'", '"'))
        except Exception:
            return raw_results  # Pass as-is if can't parse

    if not isinstance(raw_results, list):
        return str(raw_results)

    formatted_parts = []
    for i, item in enumerate(raw_results, 1):
        if isinstance(item, dict):
            title   = item.get("title", "Unknown Source")
            url     = item.get("url", "")
            content = item.get("content", "")
            score   = item.get("score", "")
            formatted_parts.append(
                f"**Source {i}: {title}**\n"
                f"URL: {url}\n"
                f"Content:\n{content}\n"
            )
        else:
            formatted_parts.append(str(item))

    return "\n---\n".join(formatted_parts)


def research_agent_node(state: AgentState):
    model_name = TASK_MODEL_MAP.get("research", "groq/llama-3.3-70b-versatile")
    llm = get_model_with_fallback(model_name)
    messages = list(state["messages"])

    if not any(isinstance(m, SystemMessage) for m in messages):
        sys_msg = SystemMessage(content=RESEARCH_SYSTEM_PROMPT)
        messages = [sys_msg] + messages

    search = TavilySearchResults(max_results=5)
    arxiv_wrapper = ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=2000)
    arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

    tools = [search, arxiv]
    llm_with_tools = llm.bind_tools(tools)

    trace = state.get("agent_trace", []) + ["research_agent"]

    # High-Resolution Self-Correction: Inject feedback if in retry loop
    feedback = state.get("eval_feedback")
    if feedback and state.get("retry_count", 0) > 0:
        messages = list(messages)
        messages.append(HumanMessage(content=(
            f"⚠️ YOUR PREVIOUS RESPONSE FAILED QUALITY AUDIT.\n"
            f"{feedback}\n"
            "Please regenerate your response and fix ALL the issues mentioned above."
        )))

    # First LLM call — decides whether to call tools
    response = llm_with_tools.invoke(messages)

    if not response.tool_calls:
        return {"messages": [response], "agent_trace": trace}

    # Execute tool calls
    new_messages = [response]
    messages.append(response)

    for tool_call in response.tool_calls:
        try:
            if tool_call["name"] == "tavily_search_results_json":
                raw_result = search.invoke(tool_call["args"])
                # Clean the Tavily result — convert dicts to readable text
                clean_result = _format_tavily_results(raw_result)
            elif tool_call["name"] == "arxiv":
                clean_result = arxiv.invoke(tool_call["args"])
            else:
                clean_result = "Unknown tool."
        except Exception as e:
            clean_result = f"Tool error: {str(e)}"

        tool_msg = ToolMessage(content=clean_result, tool_call_id=tool_call["id"])
        new_messages.append(tool_msg)
        messages.append(tool_msg)

    # Second LLM call — synthesize with explicit formatting instruction
    messages.append(HumanMessage(content=RESEARCH_SYNTHESIS_PROMPT))
    final_response = llm_with_tools.invoke(messages)
    new_messages.append(final_response)

    return {"messages": new_messages, "agent_trace": trace}
