from typing import Any
from typing_extensions import TypedDict


class Evidence(TypedDict, total=False):
    source_type: str
    source_name: str
    title: str
    url: str
    content: str
    query: str
    is_full_text: bool
    year: int
    venue: str
    citation_count: int
    paper_id: str
    relevance: float
    authority: float
    freshness: float
    completeness: float
    overall_score: float


class ResearchState(TypedDict, total=False):
    question: str
    plan: dict[str, Any]
    analysis_dimensions: list[dict[str, str]]

    queries: list[str]
    github_repos: list[str]
    github_queries: list[str]
    paper_queries: list[str]

    followup_queries: list[str]
    followup_github_repos: list[str]
    followup_github_queries: list[str]
    followup_paper_queries: list[str]

    all_evidence: list[Evidence]
    evidence: list[Evidence]

    researcher_output: dict[str, Any]
    draft_answer: str
    dimension_status: list[dict[str, str]]

    verified_claims: list[dict[str, Any]]
    claim_reviews: list[dict[str, Any]]

    critique: str
    unsupported_claims: list[Any]
    missing_perspectives: list[Any]
    sufficient: bool

    round: int
    final_answer: str
