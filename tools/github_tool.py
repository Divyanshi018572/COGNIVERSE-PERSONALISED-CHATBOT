import os
import requests
from langchain_core.tools import tool

@tool
def search_github(query: str, search_type: str = "repositories") -> str:
    """
    Searches GitHub for repositories or code.
    search_type must be either 'repositories' or 'code'.
    Use this to find reference implementations, libraries, or examples.
    """
    url = f"https://api.github.com/search/{search_type}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
        
    try:
        response = requests.get(url, headers=headers, params={"q": query, "per_page": 5}, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        items = data.get("items", [])
        if not items:
            return f"No {search_type} found on GitHub for query: {query}"
            
        results = []
        for i, item in enumerate(items):
            if search_type == "repositories":
                res = f"{i+1}. {item['full_name']} - {item.get('description', 'No description')} (Stars: {item['stargazers_count']})\nURL: {item['html_url']}"
            else:
                repo_name = item.get("repository", {}).get("full_name", "Unknown Repo")
                res = f"{i+1}. File: {item['path']} in {repo_name}\nURL: {item['html_url']}"
            results.append(res)
            
        return "\n\n".join(results)
    except Exception as e:
        return f"GitHub API Error: {str(e)}"
