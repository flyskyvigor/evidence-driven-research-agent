from typing import Any
from typing_extensions import TypedDict


class Evidence(TypedDict, total=False):
    evidence_id: str
    content_hash: str
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
    retrieved_at: str
    tool_name: str
    page: int
    page_start: int
    page_end: int
    section_title: str
    content_types: str
    parser: str
    parser_warnings: str
    ocr_used: bool
    doi: str
    parent_id: str
    matched_child_ids: list[str]
    retrieval_strategy: str
    section_prior: float


class ResearchState(TypedDict, total=False):
    session_id: str
    user_id: str
    run_id: str
    question: str
    memory_context: str
    recalled_memories: list[dict[str, Any]]
    memory_recall_mode: str
    memory_recall_error: str
    memory_write: dict[str, Any]
    memory_write_error: str
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

    tool_calls: list[dict[str, Any]]
    tool_selection_mode: str
    tool_selection_error: str
    tool_results: list[dict[str, Any]]
    tool_failures: list[dict[str, Any]]
    mcp_discovery_failures: dict[str, str]
    retrieval_history: list[dict[str, Any]]

    all_evidence: list[Evidence]
    evidence: list[Evidence]

    researcher_output: dict[str, Any]
    draft_answer: str
    dimension_status: list[dict[str, str]]

    verified_claims: list[dict[str, Any]]
    claim_reviews: list[dict[str, Any]]
    evidence_conflicts: list[dict[str, Any]]
    claim_support_metrics: list[dict[str, Any]]

    critique: str
    unsupported_claims: list[Any]
    missing_perspectives: list[Any]
    sufficient: bool

    round: int
    stop_reason: str
    approval_status: str
    pending_approval: dict[str, Any]
    run_metrics: dict[str, Any]
    final_answer: str
