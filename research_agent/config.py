"""Runtime configuration loaded from environment variables.

Configuration is validated only when a feature is initialized so that static
tools and unrelated modules remain importable without local model files.
"""

from __future__ import annotations

import os
from pathlib import Path

from research_agent.errors import ConfigurationError


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigurationError(f"Environment variable {name} is not set.")
    return value


def get_llm_model_path() -> str:
    """Return the configured local instruction-model path."""
    return _required("LLM_MODEL_PATH")


def get_embedding_model_path() -> str:
    """Return the configured local embedding-model path."""
    return _required("EMBEDDING_MODEL_PATH")


def get_reranker_model_path() -> str | None:
    """Return an optional local CrossEncoder reranker directory.

    The project never downloads a reranker implicitly.  An empty value keeps
    the deterministic dense + BM25 + RRF pipeline and records that reranking
    was not enabled.
    """
    return os.getenv("RERANKER_MODEL_PATH", "").strip() or None


def get_web_proxy() -> str | None:
    """Return the optional explicit proxy used by retrieval clients."""
    return os.getenv("WEB_PROXY", "").strip() or None


def get_github_token() -> str | None:
    """Return the optional GitHub API token."""
    return os.getenv("GITHUB_TOKEN", "").strip() or None


def get_semantic_scholar_api_key() -> str | None:
    """Return the optional Semantic Scholar API key."""
    return os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip() or None


def get_openalex_mailto() -> str | None:
    """Return the optional contact email sent to OpenAlex."""
    return os.getenv("OPENALEX_MAILTO", "").strip() or None


def get_checkpoint_db_path() -> Path:
    raw = os.getenv("CHECKPOINT_DB_PATH", "").strip()
    return Path(raw).expanduser() if raw else PROJECT_ROOT / "data" / "checkpoints.sqlite"


def get_memory_db_path() -> Path:
    raw = os.getenv("MEMORY_DB_PATH", "").strip()
    return Path(raw).expanduser() if raw else PROJECT_ROOT / "data" / "memory.sqlite"


def get_require_plan_approval() -> bool:
    return os.getenv("REQUIRE_PLAN_APPROVAL", "false").strip().lower() in {
        "1", "true", "yes", "on"
    }


def get_max_context_chars() -> int:
    raw = os.getenv("MAX_CONTEXT_CHARS", "12000").strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError("MAX_CONTEXT_CHARS must be an integer.") from exc
    if not 2000 <= value <= 100_000:
        raise ConfigurationError("MAX_CONTEXT_CHARS must be between 2000 and 100000.")
    return value


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer.") from exc
    if not minimum <= value <= maximum:
        raise ConfigurationError(
            f"{name} must be between {minimum} and {maximum}."
        )
    return value


def get_rag_parent_chunk_size() -> int:
    return _bounded_int("RAG_PARENT_CHUNK_SIZE", 1800, 600, 8000)


def get_rag_parent_chunk_overlap() -> int:
    return _bounded_int("RAG_PARENT_CHUNK_OVERLAP", 200, 0, 2000)


def get_rag_child_chunk_size() -> int:
    return _bounded_int("RAG_CHILD_CHUNK_SIZE", 450, 100, 2000)


def get_rag_child_chunk_overlap() -> int:
    return _bounded_int("RAG_CHILD_CHUNK_OVERLAP", 80, 0, 500)


def get_rag_rrf_k() -> int:
    return _bounded_int("RAG_RRF_K", 60, 1, 500)


def get_memory_recall_limit() -> int:
    return _bounded_int("MEMORY_RECALL_LIMIT", 5, 1, 20)


def get_memory_ttl_days() -> int:
    return _bounded_int("MEMORY_TTL_DAYS", 180, 1, 3650)


def get_pdf_extract_tables() -> bool:
    return os.getenv("PDF_EXTRACT_TABLES", "true").strip().lower() in {
        "1", "true", "yes", "on"
    }


def get_pdf_min_text_chars() -> int:
    return _bounded_int("PDF_MIN_TEXT_CHARS", 80, 0, 5000)


def get_pdf_max_pages() -> int:
    return _bounded_int("PDF_MAX_PAGES", 500, 1, 10000)


def get_pdf_max_file_mb() -> int:
    return _bounded_int("PDF_MAX_FILE_MB", 200, 1, 5000)


def get_pdf_ocr_mode() -> str:
    value = os.getenv("PDF_OCR_MODE", "disabled").strip().lower()
    if value not in {"disabled", "tesseract"}:
        raise ConfigurationError("PDF_OCR_MODE must be disabled or tesseract.")
    return value


def get_pdf_ocr_language() -> str:
    value = os.getenv("PDF_OCR_LANGUAGE", "eng").strip()
    if not value or len(value) > 100 or not all(
        character.isalnum() or character in {"_", "+", "-"}
        for character in value
    ):
        raise ConfigurationError("PDF_OCR_LANGUAGE contains invalid characters.")
    return value
