import os
import requests
import json
import re
import logging
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from utils.logger import get_logger
from core.prompts import FORMATTING_DIRECTIVE

logger = get_logger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_headers() -> dict:
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def _extract_owner_repo(repo_url: str) -> tuple:
    match = re.search(r"github\.com/([^/\s]+)/([^/\s]+)", repo_url)
    if match:
        owner = match.group(1)
        repo = match.group(2).replace(".git", "").strip()
        repo = repo.rstrip(".,;:'\"()[]{}?!")
        return owner, repo
    parts = repo_url.strip("/ ").split("/")
    if len(parts) >= 2:
        owner = parts[-2]
        repo = parts[-1].replace(".git", "").strip().rstrip(".,;:'\"()[]{}?!")
        return owner, repo
    return "", ""


def fetch_repo_metadata(owner: str, repo: str) -> dict:
    try:
        url = f"https://api.github.com/repos/{owner}/{repo}"
        resp = requests.get(url, headers=_get_headers(), timeout=10)
        if resp.status_code == 200:
            d = resp.json()
            return {
                "name": d.get("full_name", f"{owner}/{repo}"),
                "description": d.get("description") or "No description provided.",
                "language": d.get("language") or "Unknown",
                "stars": d.get("stargazers_count", 0),
                "forks": d.get("forks_count", 0),
                "open_issues": d.get("open_issues_count", 0),
                "created_at": (d.get("created_at") or "")[:10],
                "updated_at": (d.get("updated_at") or "")[:10],
                "clone_url": d.get("clone_url", f"https://github.com/{owner}/{repo}.git"),
                "topics": ", ".join(d.get("topics", [])) or "None",
            }
        elif resp.status_code == 404:
            return {"error": "Repository not found (404). It may be private or the URL may be wrong."}
        elif resp.status_code == 403:
            return {"error": "GitHub API rate limit exceeded. Add GITHUB_TOKEN to .env."}
        return {"error": f"GitHub returned status {resp.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def fetch_file(owner: str, repo: str, path: str = "README.md") -> str:
    for branch in ["main", "master"]:
        try:
            url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                text = resp.text
                return text[:15000] + "\n...[TRUNCATED]..." if len(text) > 15000 else text
        except Exception:
            pass
    return f"[Could not fetch {path} — not found in main or master branch]"


def search_repos(topic: str) -> list:
    try:
        url = f"https://api.github.com/search/repositories?q={requests.utils.quote(topic)}&sort=updated&order=desc&per_page=8"
        resp = requests.get(url, headers=_get_headers(), timeout=10)
        if resp.status_code == 200:
            items = resp.json().get("items", [])[:6]
            return [{
                "name": i.get("full_name"),
                "description": (i.get("description") or "N/A")[:100],
                "stars": i.get("stargazers_count", 0),
                "open_issues": i.get("open_issues_count", 0),
                "language": i.get("language") or "N/A",
                "last_updated": (i.get("updated_at") or "")[:10],
            } for i in items]
    except Exception as e:
        logger.error("github_search_failed", error=str(e))
    return []


# ── Agent Node ────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = f"""{FORMATTING_DIRECTIVE}

You are the Cognibot GitHub Architecture Agent. You will be provided with pre-fetched GitHub data.
Your job is to synthesize that data into a rich analysis.

### OUTPUT REQUIREMENTS (follow ALL of these):
1. **Repository Overview** - bold title, description, language, stars, forks, open issues.
2. **Tech Stack Table** - a Markdown table listing all technologies/tools you can identify from the README.
3. **Architecture Flowchart** - a Mermaid.js `graph TD` flowchart showing how the main components connect.
4. **How to Contribute** - bullet points extracted from CONTRIBUTING.md or the README.
5. **Clone Command** - a bash code block with the exact `git clone` command.

Keep headings clear. Do NOT skip any section.
"""

def _detect_intent(user_msg: str) -> str:
    """Detect whether this is a repo analysis or topic search."""
    msg_lower = user_msg.lower()
    if "github.com" in msg_lower or "analyze" in msg_lower:
        return "analyze"
    if "find top" in msg_lower or "topic" in msg_lower or "search" in msg_lower:
        return "search"
    return "analyze"


def _extract_repo_url(text: str) -> str:
    match = re.search(r"https?://github\.com/[^\s\)\"']+", text)
    return match.group(0) if match else ""


def _extract_topic(text: str) -> str:
    patterns = [
        r"topic ['\"]?([a-zA-Z0-9_\-]+)['\"]?",
        r"for ['\"]?([a-zA-Z0-9_\-]+)['\"]?",
        r"about ['\"]?([a-zA-Z0-9_\-]+)['\"]?",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1)
    words = text.split()
    return words[-1] if words else "machine-learning"


def github_agent_node(state: dict) -> dict:
    """Fetch GitHub data directly, then feed it to the LLM for rich analysis."""
    messages = state.get("messages", [])
    user_msg = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            content = m.content
            if isinstance(content, list):
                user_msg = next((x.get("text","") for x in content if x.get("type")=="text"), "")
            else:
                user_msg = content
            break

    logger.info("github_agent_started", user_msg=user_msg[:100])

    intent = _detect_intent(user_msg)
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

    # ── Branch 1: Repo Analysis ───────────────────────────────────────────────
    if intent == "analyze":
        repo_url = _extract_repo_url(user_msg)
        if not repo_url:
            # Try to get from message
            repo_url = user_msg.strip()

        owner, repo = _extract_owner_repo(repo_url)
        logger.info("github_analyzing", owner=owner, repo=repo)

        if not owner or not repo:
            ai_msg = AIMessage(content=f"❌ I couldn't extract a valid GitHub repository URL from your message. Please provide a full URL like `https://github.com/owner/repo`.")
            return {"messages": [ai_msg], "agent_trace": state.get("agent_trace", []) + ["github_agent"]}

        # Fetch data in parallel-style (sequential is fine, it's fast)
        meta = fetch_repo_metadata(owner, repo)
        readme = fetch_file(owner, repo, "README.md")
        contributing = fetch_file(owner, repo, "CONTRIBUTING.md")

        if "error" in meta:
            ai_msg = AIMessage(content=f"❌ GitHub API Error: {meta['error']}")
            return {"messages": [ai_msg], "agent_trace": state.get("agent_trace", []) + ["github_agent"]}

        # Build a rich context prompt
        context = f"""
## Repository: {meta['name']}
- **Description**: {meta['description']}
- **Primary Language**: {meta['language']}
- **Stars**: {meta['stars']} | **Forks**: {meta['forks']} | **Open Issues**: {meta['open_issues']}
- **Topics**: {meta['topics']}
- **Created**: {meta['created_at']} | **Last Updated**: {meta['updated_at']}
- **Clone URL**: {meta['clone_url']}

## README.md Content:
{readme}

## CONTRIBUTING.md Content:
{contributing}
"""
        logger.info("github_context_built", repo=f"{owner}/{repo}", readme_len=len(readme))

        prompt_messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=f"Analyze the repository `{owner}/{repo}` using this data:\n\n{context}")
        ]

        # High-Resolution Self-Correction: Inject feedback if we are in a retry loop
        feedback = state.get("eval_feedback")
        if feedback and state.get("retry_count", 0) > 0:
            prompt_messages.append(HumanMessage(content=(
                f"⚠️ YOUR PREVIOUS RESPONSE FAILED QUALITY AUDIT.\n"
                f"{feedback}\n"
                "Please regenerate your response and fix ALL the issues mentioned above."
            )))

        try:
            response = llm.invoke(prompt_messages)
            content = response.content
            logger.info("github_llm_response", content_len=len(content) if content else 0)
            if not content:
                content = f"# {meta['name']}\n\n**Description**: {meta['description']}\n\n**Stars**: {meta['stars']} | **Forks**: {meta['forks']}\n\n**Clone**: `git clone {meta['clone_url']}`"
        except Exception as e:
            logger.error("github_llm_failed", error=str(e))
            content = f"❌ LLM error: {e}"

        ai_msg = AIMessage(content=content)
        return {"messages": [ai_msg], "agent_trace": state.get("agent_trace", []) + ["github_agent"]}

    # ── Branch 2: Topic Search ────────────────────────────────────────────────
    else:
        topic = _extract_topic(user_msg)
        logger.info("github_searching_topic", topic=topic)
        repos = search_repos(topic)

        if not repos:
            ai_msg = AIMessage(content=f"❌ No repositories found for topic `{topic}`. Try a different keyword.")
            return {"messages": [ai_msg], "agent_trace": state.get("agent_trace", []) + ["github_agent"]}

        repos_json = json.dumps(repos, indent=2)
        prompt_messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Here are the top GitHub repositories for the topic **'{topic}'**, "
                f"ranked by recent activity (not just stars):\n\n```json\n{repos_json}\n```\n\n"
                "Present this as a **Markdown Table** with columns: Rank, Repository, Language, Stars, Open Issues, Last Updated, Description. "
                "Then add a brief summary of which repository would be best to contribute to and why."
            ))
        ]

        try:
            response = llm.invoke(prompt_messages)
            content = response.content
        except Exception as e:
            content = f"❌ LLM error: {e}"

        ai_msg = AIMessage(content=content)
        return {"messages": [ai_msg], "agent_trace": state.get("agent_trace", []) + ["github_agent"]}
