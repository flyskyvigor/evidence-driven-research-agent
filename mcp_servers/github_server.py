import httpx
from mcp.server import MCPServer

from research_agent.config import get_github_token, get_web_proxy


mcp = MCPServer(
    "ReasoningAgent GitHub Server",
    instructions="Provide structured GitHub repository evidence."
)

API_BASE = "https://api.github.com"


def _headers(accept="application/vnd.github+json"):
    headers = {
        "Accept": accept,
        "User-Agent": "ReasoningAgent/1.0"
    }

    token = get_github_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    return headers


def _client():
    return httpx.Client(
        proxy=get_web_proxy(),
        timeout=20,
        follow_redirects=True,
        trust_env=False
    )


def _latest_release(client, full_name):
    try:
        response = client.get(
            f"{API_BASE}/repos/{full_name}/releases/latest",
            headers=_headers()
        )

        if response.status_code != 200:
            return {}

        data = response.json()

        return {
            "tag": data.get("tag_name", ""),
            "published_at": data.get("published_at", ""),
            "url": data.get("html_url", "")
        }
    except Exception:
        return {}


def _readme(client, full_name):
    try:
        response = client.get(
            f"{API_BASE}/repos/{full_name}/readme",
            headers=_headers(
                "application/vnd.github.raw+json"
            )
        )

        if response.status_code != 200:
            return ""

        return response.text[:5000]
    except Exception:
        return ""


def _build_repository(client, item):
    full_name = item.get("full_name", "")

    return {
        "full_name": full_name,
        "url": item.get("html_url", ""),
        "description": item.get("description") or "",
        "stars": item.get("stargazers_count", 0),
        "forks": item.get("forks_count", 0),
        "open_issues": item.get("open_issues_count", 0),
        "language": item.get("language") or "",
        "archived": item.get("archived", False),
        "created_at": item.get("created_at", ""),
        "updated_at": item.get("updated_at", ""),
        "pushed_at": item.get("pushed_at", ""),
        "topics": item.get("topics", []),
        "latest_release": _latest_release(
            client,
            full_name
        ),
        "readme": _readme(
            client,
            full_name
        )
    }


@mcp.tool()
def get_repository(full_name: str) -> dict:
    """Get an exact public GitHub repository."""
    try:
        with _client() as client:
            response = client.get(
                f"{API_BASE}/repos/{full_name}",
                headers=_headers()
            )

            if response.status_code != 200:
                return {}

            return _build_repository(
                client,
                response.json()
            )
    except Exception:
        return {}


@mcp.tool()
def search_repositories(
    query: str,
    max_results: int = 1
) -> list[dict]:
    """Search public GitHub repositories."""
    if not isinstance(query, str) or not query.strip():
        return []

    max_results = max(
        1,
        min(int(max_results), 5)
    )

    try:
        with _client() as client:
            response = client.get(
                f"{API_BASE}/search/repositories",
                params={
                    "q": query.strip(),
                    "sort": "stars",
                    "order": "desc",
                    "per_page": max_results
                },
                headers=_headers()
            )

            if response.status_code != 200:
                return []

            repositories = []

            for item in response.json().get("items", []):
                repositories.append(
                    _build_repository(
                        client,
                        item
                    )
                )

            return repositories
    except Exception:
        return []


if __name__ == "__main__":
    mcp.run(transport="stdio")
