from langchain_experimental.tools import PythonREPLTool
from langchain_community.agent_toolkits import FileManagementToolkit
from tools.github_tool import search_github

def get_coding_tools():
    repl_tool = PythonREPLTool()
    file_toolkit = FileManagementToolkit(root_dir=".")
    file_tools = file_toolkit.get_tools()
    
    return [repl_tool, search_github] + list(file_tools)
