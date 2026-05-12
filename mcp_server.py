from mcp.server.fastmcp import FastMCP
import httpx
import os

# Create an MCP server for Cognibot
mcp = FastMCP("cognibot-mcp-server")

@mcp.tool()
async def analyze_github_repo(repo_url: str) -> str:
    """
    Analyzes a GitHub repository and returns its architecture and tech stack.
    Use this when you need deep insights into a specific repository.
    """
    # In a real MCP implementation, this would call the internal github_agent logic
    # For now, we'll provide a helpful description of how to use the existing tool
    return f"To analyze {repo_url}, use the GitHub Integration sidebar in Cognibot or ask the GitHub Agent directly."

@mcp.resource("config://app-info")
def get_app_info() -> str:
    """Provides metadata about the Cognibot application."""
    return """
    Cognibot is a multi-agent AI system built with LangGraph and FastAPI.
    It features specialized agents for:
    - GitHub Analysis
    - Coding & Scripting
    - Research & Web Search
    - RAG (Document Analysis)
    - Memory & Personalization
    """

if __name__ == "__main__":
    mcp.run()
