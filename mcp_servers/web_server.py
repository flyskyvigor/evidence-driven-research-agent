import json
import logging
import ipaddress
import socket
from urllib.parse import urljoin, urlparse

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
        and not parsed.username
        and not parsed.password
    )


def _is_public_destination(url):
    """拒绝 loopback、私网、链路本地和云元数据常用地址。"""
    hostname = (urlparse(url).hostname or "").strip().lower()
    if not hostname or hostname in {"localhost", "metadata.google.internal"}:
        return False
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(hostname, None)}
    except socket.gaierror:
        return False
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            return False
    return True

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
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    try:
        max_results = max(1, min(int(max_results), 10))
    except (TypeError, ValueError) as exc:
        raise ValueError("max_results must be an integer") from exc
    proxy = get_web_proxy()

    try:
        results = DDGS(proxy=proxy, timeout=15).text(query.strip(), max_results=max_results)
    except Exception as exc:
        raise RuntimeError(f"web_search_failed:{type(exc).__name__}") from exc

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

    if not _is_public_destination(url):
        result["error"] = "non_public_destination"
        return result

    if _should_skip_url(url):
        result["error"] = "unsupported_file_type"
        return result

    try:
        with httpx.Client(
            proxy=get_web_proxy(),
            timeout=15,
            follow_redirects=False,
            trust_env=False,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/120 Safari/537.36"
                )
            }
        ) as client:
            current_url = url
            response = None
            for _ in range(6):
                response = client.get(current_url)
                if not response.is_redirect:
                    break
                location = response.headers.get("location", "")
                next_url = urljoin(current_url, location)
                if not _valid_http_url(next_url) or not _is_public_destination(next_url):
                    result["error"] = "redirected_to_non_public_destination"
                    return result
                current_url = next_url
            else:
                result["error"] = "too_many_redirects"
                return result

        if response is None:
            result["error"] = "no_response"
            return result

        if response.status_code != 200:
            result["error"] = (
                f"http_status_{response.status_code}"
            )
            return result

        final_url = str(response.url)

        result["url"] = final_url

        if not _is_public_destination(final_url):
            result["error"] = "redirected_to_non_public_destination"
            return result

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
