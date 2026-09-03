"""Runtime configuration loaded from environment variables.

Configuration is validated only when a feature is initialized so that static
tools and unrelated modules remain importable without local model files.
"""

from __future__ import annotations

import os


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Environment variable {name} is not set.")
    return value


def get_llm_model_path() -> str:
    """Return the configured local instruction-model path."""
    return _required("LLM_MODEL_PATH")


def get_embedding_model_path() -> str:
    """Return the configured local embedding-model path."""
    return _required("EMBEDDING_MODEL_PATH")


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
