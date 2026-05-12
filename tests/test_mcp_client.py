import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_mcp_server():
    print(">>> Initializing MCP Client...")
    
    # Configure the client to connect to your mcp_server.py via standard input/output
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the connection
                await session.initialize()
                print("[SUCCESS] Connected to FastMCP Server!\n")
                
                # 1. Check Resources
                resources = await session.list_resources()
                print("--- AVAILABLE RESOURCES ---")
                for res in resources.resources:
                    print(f"  - {res.uri} ({res.description})")
                    
                # 2. Check Tools
                tools = await session.list_tools()
                print("\n--- AVAILABLE TOOLS ---")
                for tool in tools.tools:
                    print(f"  - {tool.name}: {tool.description}")
                    
                # 3. Test Tool Execution
                print("\n--- TESTING TOOL EXECUTION (analyze_github_repo) ---")
                result = await session.call_tool(
                    "analyze_github_repo", 
                    arguments={"repo_url": "https://github.com/test/repo"}
                )
                print(f"  Response: {result.content[0].text}")
                
    except Exception as e:
        print(f"[ERROR] Error connecting to MCP server: {e}")

if __name__ == "__main__":
    asyncio.run(test_mcp_server())
