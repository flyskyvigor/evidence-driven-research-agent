import xml.etree.ElementTree as ET

import httpx
from mcp.server import MCPServer

from research_agent.config import (
    get_openalex_mailto,
    get_semantic_scholar_api_key,
    get_web_proxy,
)


mcp = MCPServer(
    "ReasoningAgent Paper Server",
    instructions="Provide structured academic paper evidence."
)

S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
OPENALEX_API = "https://api.openalex.org/works"
ARXIV_API = "https://export.arxiv.org/api/query"


def _client():
    return httpx.Client(
        proxy=get_web_proxy(),
        timeout=25,
        follow_redirects=True,
        trust_env=False,
        headers={"User-Agent": "ReasoningAgent/1.0"}
    )


def _normalize_text(text):
    return " ".join((text or "").split())


def _contains_cjk(text):
    return any(
        "\u4e00" <= char <= "\u9fff"
        for char in (text or "")
    )


def _restore_openalex_abstract(inverted_index):
    if not isinstance(inverted_index, dict):
        return ""

    positioned_words = []

    for word, positions in inverted_index.items():
        if not isinstance(word, str) or not isinstance(positions, list):
            continue

        for position in positions:
            if isinstance(position, int) and position >= 0:
                positioned_words.append((position, word))

    positioned_words.sort(key=lambda item: item[0])
    return _normalize_text(
        " ".join(word for _, word in positioned_words)
    )


def _strip_identifier_url(value, prefix):
    value = (value or "").strip()
    if value.lower().startswith(prefix.lower()):
        return value[len(prefix):]
    return value


def _search_semantic_scholar(client, query, max_results):
    api_key = get_semantic_scholar_api_key()
    if not api_key:
        return []

    fields = (
        "title,url,abstract,year,venue,authors,citationCount,"
        "publicationDate,openAccessPdf,externalIds"
    )

    response = client.get(
        f"{S2_API_BASE}/paper/search",
        params={
            "query": query,
            "limit": max_results,
            "fields": fields
        },
        headers={"x-api-key": api_key}
    )

    if response.status_code != 200:
        return []

    papers = []

    for item in response.json().get("data", []):
        pdf = item.get("openAccessPdf") or {}
        external_ids = item.get("externalIds") or {}

        papers.append({
            "provider": "Semantic Scholar",
            "paper_id": item.get("paperId", ""),
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "abstract": item.get("abstract") or "",
            "year": item.get("year"),
            "publication_date": item.get("publicationDate") or "",
            "venue": item.get("venue") or "",
            "authors": [
                author.get("name", "")
                for author in item.get("authors", [])
            ],
            "citation_count": item.get("citationCount", 0),
            "pdf_url": pdf.get("url", ""),
            "arxiv_id": external_ids.get("ArXiv", ""),
            "doi": external_ids.get("DOI", "")
        })

    return papers


def _search_openalex(client, query, max_results):
    params = {
        "search": query,
        "per_page": max_results
    }

    mailto = get_openalex_mailto()
    if mailto:
        params["mailto"] = mailto

    response = client.get(
        OPENALEX_API,
        params=params
    )

    if response.status_code != 200:
        return []

    papers = []

    for item in response.json().get("results", []):
        primary_location = item.get("primary_location") or {}
        best_oa_location = item.get("best_oa_location") or {}
        source = primary_location.get("source") or {}
        ids = item.get("ids") or {}

        openalex_url = item.get("id") or ids.get("openalex") or ""
        paper_id = openalex_url.rstrip("/").split("/")[-1]

        doi = item.get("doi") or ids.get("doi") or ""
        doi = _strip_identifier_url(doi, "https://doi.org/")

        arxiv_id = ids.get("arxiv") or ""
        arxiv_id = _strip_identifier_url(
            arxiv_id,
            "https://arxiv.org/abs/"
        )

        url = (
            primary_location.get("landing_page_url")
            or (f"https://doi.org/{doi}" if doi else "")
            or openalex_url
        )

        authors = []
        for authorship in item.get("authorships") or []:
            author = authorship.get("author") or {}
            name = _normalize_text(author.get("display_name"))
            if name:
                authors.append(name)

        papers.append({
            "provider": "OpenAlex",
            "paper_id": paper_id,
            "title": item.get("display_name") or item.get("title") or "",
            "url": url,
            "abstract": _restore_openalex_abstract(
                item.get("abstract_inverted_index")
            ),
            "year": item.get("publication_year"),
            "publication_date": item.get("publication_date") or "",
            "venue": source.get("display_name") or "",
            "authors": authors,
            "citation_count": item.get("cited_by_count") or 0,
            "pdf_url": (
                best_oa_location.get("pdf_url")
                or primary_location.get("pdf_url")
                or ""
            ),
            "arxiv_id": arxiv_id,
            "doi": doi
        })

    return papers


def _search_arxiv(client, query, max_results):
    response = client.get(
        ARXIV_API,
        params={
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
    )

    if response.status_code != 200:
        return []

    root = ET.fromstring(response.text)

    atom = "http://www.w3.org/2005/Atom"
    arxiv_ns = "http://arxiv.org/schemas/atom"

    papers = []

    for entry in root.findall(f"{{{atom}}}entry"):
        title = _normalize_text(
            entry.findtext(f"{{{atom}}}title")
        )
        abstract = _normalize_text(
            entry.findtext(f"{{{atom}}}summary")
        )
        url = (
            entry.findtext(f"{{{atom}}}id") or ""
        ).strip()

        published = (
            entry.findtext(f"{{{atom}}}published") or ""
        ).strip()

        authors = [
            _normalize_text(
                author.findtext(f"{{{atom}}}name")
            )
            for author in entry.findall(f"{{{atom}}}author")
        ]

        pdf_url = ""

        for link in entry.findall(f"{{{atom}}}link"):
            if link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib.get("href", "")
                break

        arxiv_id = url.rstrip("/").split("/")[-1]
        doi = entry.findtext(f"{{{arxiv_ns}}}doi") or ""

        year = None
        if len(published) >= 4 and published[:4].isdigit():
            year = int(published[:4])

        papers.append({
            "provider": "arXiv",
            "paper_id": f"ARXIV:{arxiv_id}",
            "title": title,
            "url": url,
            "abstract": abstract,
            "year": year,
            "publication_date": published[:10],
            "venue": "arXiv",
            "authors": authors,
            "citation_count": 0,
            "pdf_url": pdf_url,
            "arxiv_id": arxiv_id,
            "doi": doi
        })

    return papers


@mcp.tool()
def search_papers(query: str, max_results: int = 4) -> list[dict]:
    """Search papers using sequential provider fallbacks."""
    if not isinstance(query, str) or not query.strip():
        return []

    query = query.strip()
    try:
        max_results = max(1, min(int(max_results), 10))
    except (TypeError, ValueError):
        return []

    try:
        with _client() as client:
            providers = []

            if get_semantic_scholar_api_key():
                providers.append(_search_semantic_scholar)

            providers.append(_search_openalex)

            # arXiv is primarily an English-paper environment. Avoid spending
            # its limited request budget on Chinese queries with low recall.
            if not _contains_cjk(query):
                providers.append(_search_arxiv)

            papers = []
            seen = set()

            for provider in providers:
                try:
                    results = provider(
                        client,
                        query,
                        max_results
                    )
                except Exception:
                    continue

                if not isinstance(results, list):
                    continue

                for paper in results:
                    if not isinstance(paper, dict):
                        continue

                    doi = (paper.get("doi") or "").strip().lower()
                    title = _normalize_text(
                        paper.get("title")
                    ).lower()
                    identity = ("doi", doi) if doi else ("title", title)

                    if not title or identity in seen:
                        continue

                    seen.add(identity)
                    papers.append(paper)

                    if len(papers) >= max_results:
                        return papers[:max_results]

            return papers[:max_results]
    except Exception:
        return []


@mcp.tool()
def get_paper(paper_id: str) -> dict:
    """Get detailed Semantic Scholar paper metadata when an API key is available."""
    api_key = get_semantic_scholar_api_key()

    if not api_key:
        return {}

    fields = (
        "title,url,abstract,year,venue,authors,citationCount,"
        "publicationDate,openAccessPdf,externalIds"
    )

    try:
        with _client() as client:
            response = client.get(
                f"{S2_API_BASE}/paper/{paper_id}",
                params={"fields": fields},
                headers={"x-api-key": api_key}
            )

            if response.status_code != 200:
                return {}

            item = response.json()
            pdf = item.get("openAccessPdf") or {}
            external_ids = item.get("externalIds") or {}

            return {
                "provider": "Semantic Scholar",
                "paper_id": item.get("paperId", ""),
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "abstract": item.get("abstract") or "",
                "year": item.get("year"),
                "publication_date": item.get("publicationDate") or "",
                "venue": item.get("venue") or "",
                "authors": [
                    author.get("name", "")
                    for author in item.get("authors", [])
                ],
                "citation_count": item.get("citationCount", 0),
                "pdf_url": pdf.get("url", ""),
                "arxiv_id": external_ids.get("ArXiv", ""),
                "doi": external_ids.get("DOI", "")
            }
    except Exception:
        return {}


if __name__ == "__main__":
    mcp.run(transport="stdio")
