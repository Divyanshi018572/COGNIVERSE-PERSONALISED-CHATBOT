from langgraph.prebuilt import ToolNode
from agents.tools import get_coding_tools

coding_tool_node = ToolNode(get_coding_tools())
