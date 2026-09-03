import json
import logging
from urllib.parse import urlparse

import httpx
from ddgs import DDGS
from mcp.server import MCPServer
from trafilatura import extract

from research_agent.config import get_web_proxy

SKIP_SUFFIXES = {
    ".pdf",
    ".xml",
    ".rss",
    ".atom",
    ".zip",
    ".gz",
    ".tar",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx"
}

logging.getLogger("ddgs").setLevel(logging.WARNING)
logging.getLogger("trafilatura").setLevel(logging.CRITICAL)
logging.getLogger("trafilatura.core").setLevel(logging.CRITICAL)
logging.getLogger("trafilatura.utils").setLevel(logging.CRITICAL)

mcp = MCPServer(
    "ReasoningAgent Web Server",
    instructions="Provide web search and webpage extraction for evidence-driven research."
)

def _valid_http_url(url):
    if not isinstance(url, str):
        return False

    url = url.strip()

    if not url:
        return False

    parsed = urlparse(url)

    return (
        parsed.scheme in {"http", "https"}
        and bool(parsed.netloc)
    )

def _should_skip_url(url):
    path = urlparse(url).path.lower()

    return any(
        path.endswith(suffix)
        for suffix in SKIP_SUFFIXES
    )

def _is_html_response(response):
    content_type = (
        response.headers
        .get("content-type", "")
        .lower()
    )

    return (
        "text/html" in content_type
        or "application/xhtml+xml" in content_type
    )

@mcp.tool()
def web_search(query: str, max_results: int = 4) -> list[dict[str, str]]:
    """Search the web and return candidate evidence."""
    proxy = get_web_proxy()

    try:
        results = DDGS(proxy=proxy, timeout=15).text(query, max_results=max_results)
    except Exception:
        return []

    return [{
        "title": item.get("title", ""),
        "url": item.get("href", ""),
        "content": item.get("body", "")
    } for item in results]


@mcp.tool()
def fetch_page(url: str) -> dict:
    """Fetch and extract readable text from a normal HTML webpage."""

    result = {
        "url": url if isinstance(url, str) else "",
        "title": "",
        "content": "",
        "site_name": "",
        "error": ""
    }

    if not _valid_http_url(url):
        result["error"] = "invalid_url"
        return result

    url = url.strip()

    if _should_skip_url(url):
        result["error"] = "unsupported_file_type"
        return result

    try:
        with httpx.Client(
            proxy=get_web_proxy(),
            timeout=15,
            follow_redirects=True,
            trust_env=False,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/120 Safari/537.36"
                )
            }
        ) as client:
            response = client.get(url)

        if response.status_code != 200:
            result["error"] = (
                f"http_status_{response.status_code}"
            )
            return result

        final_url = str(response.url)

        result["url"] = final_url

        if _should_skip_url(final_url):
            result["error"] = "unsupported_file_type"
            return result

        if not _is_html_response(response):
            content_type = response.headers.get(
                "content-type",
                ""
            )

            result["error"] = (
                f"unsupported_content_type:{content_type}"
            )
            return result

        html = response.text

        if not html or len(html.strip()) < 100:
            result["error"] = "empty_html"
            return result

        extracted = extract(
            html,
            url=final_url,
            output_format="json",
            with_metadata=True,
            include_comments=False
        )

        if not extracted:
            result["error"] = "extraction_failed"
            return result

        data = json.loads(extracted)

        content = (
            data.get("text")
            or data.get("raw_text")
            or ""
        ).strip()

        if not content:
            result["error"] = "empty_extracted_content"
            return result

        result.update({
            "title": data.get("title") or "",
            "content": content[:6000],
            "site_name": (
                data.get("sitename")
                or urlparse(final_url).netloc
            ),
            "error": ""
        })

        return result

    except Exception as e:
        result["error"] = (
            f"{type(e).__name__}: {str(e)[:200]}"
        )
        return result

if __name__ == "__main__":
    mcp.run(transport="stdio")
