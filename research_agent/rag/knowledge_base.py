"""Versioned parent-child hybrid index for the local knowledge base.

Small child chunks are embedded and indexed because they improve recall. A
matched child resolves back to its larger parent chunk before entering the
Evidence pipeline, preserving enough context for Claim verification. Dense
and BM25 rankings are combined with RRF; an optional local CrossEncoder can
rerank the fused candidates.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from research_agent.config import (
    get_embedding_model_path,
    get_pdf_extract_tables,
    get_pdf_max_file_mb,
    get_pdf_max_pages,
    get_pdf_min_text_chars,
    get_pdf_ocr_language,
    get_pdf_ocr_mode,
    get_rag_child_chunk_overlap,
    get_rag_child_chunk_size,
    get_rag_parent_chunk_overlap,
    get_rag_parent_chunk_size,
    get_rag_rrf_k,
    get_reranker_model_path,
)
from research_agent.errors import DocumentParseError
from research_agent.rag.chunking import split_structured_document
from research_agent.rag.hybrid import BM25Index, reciprocal_rank_fusion
from research_agent.rag.pdf_parser import AcademicPDFLoader, PARSER_NAME
from research_agent.rag.reranker import LocalCrossEncoderReranker


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_DIR = PROJECT_ROOT / "data" / "knowledge"
INDEX_DIR = PROJECT_ROOT / "data" / "index"
INDEX_MODEL_FILE = INDEX_DIR / "embedding_model.txt"
INDEX_MANIFEST_FILE = INDEX_DIR / "manifest.json"
PARENT_STORE_FILE = INDEX_DIR / "parents.jsonl"
CHILD_STORE_FILE = INDEX_DIR / "children.jsonl"
INDEX_SCHEMA_VERSION = 3


def create_embeddings():
    return HuggingFaceEmbeddings(
        model_name=get_embedding_model_path(),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_documents() -> list[Document]:
    documents: list[Document] = []
    for path in sorted(KNOWLEDGE_DIR.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        try:
            if suffix == ".pdf":
                docs = AcademicPDFLoader(path).load()
            elif suffix in {".txt", ".md"}:
                docs = TextLoader(
                    str(path), encoding="utf-8", autodetect_encoding=True
                ).load()
            else:
                continue
        except DocumentParseError:
            logger.exception("knowledge_pdf_parse_failed file=%s", path.name)
            raise
        except Exception as exc:
            logger.warning("knowledge_document_skipped file=%s error=%s", path.name, exc)
            continue

        for doc in docs:
            doc.metadata["file_name"] = path.name
            doc.metadata["file_path"] = str(path)
        documents.extend(docs)
    return documents


def _stable_id(prefix: str, *parts: Any) -> str:
    payload = "\u241f".join(str(part or "") for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()
    return f"{prefix}-{digest[:20]}"


def _json_safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            safe[str(key)] = value
        else:
            safe[str(key)] = str(value)
    return safe


def _source_key(document: Document) -> str:
    metadata = document.metadata
    name = metadata.get("file_name") or Path(
        str(metadata.get("source") or "unknown")
    ).name
    page = metadata.get("page_start") or metadata.get("page_number")
    if page is None:
        page = int(metadata.get("page", 0) or 0) + 1
    return f"{name}:page:{page}"


def create_parent_child_chunks(
    documents: list[Document],
    *,
    parent_chunk_size: int | None = None,
    parent_chunk_overlap: int | None = None,
    child_chunk_size: int | None = None,
    child_chunk_overlap: int | None = None,
) -> tuple[list[Document], list[Document]]:
    """Split source pages into retrievable children and contextual parents."""
    parent_size = parent_chunk_size or get_rag_parent_chunk_size()
    parent_overlap = (
        get_rag_parent_chunk_overlap()
        if parent_chunk_overlap is None else parent_chunk_overlap
    )
    child_size = child_chunk_size or get_rag_child_chunk_size()
    child_overlap = (
        get_rag_child_chunk_overlap()
        if child_chunk_overlap is None else child_chunk_overlap
    )
    if child_size >= parent_size:
        raise ValueError("RAG_CHILD_CHUNK_SIZE must be smaller than parent size")
    if parent_overlap >= parent_size or child_overlap >= child_size:
        raise ValueError("chunk overlap must be smaller than chunk size")

    parents: list[Document] = []
    children: list[Document] = []
    for source_document in documents:
        source_key = _source_key(source_document)
        source_id = _stable_id("SRC", source_key, source_document.page_content)
        source_metadata = _json_safe_metadata(source_document.metadata)
        source_metadata["source_id"] = source_id
        source_metadata["source_key"] = source_key
        parent_parts = split_structured_document(
            Document(page_content=source_document.page_content, metadata=source_metadata),
            chunk_size=parent_size,
            chunk_overlap=parent_overlap,
        )
        for parent_index, parent in enumerate(parent_parts):
            parent_id = _stable_id("PAR", source_id, parent_index, parent.page_content)
            parent_metadata = dict(parent.metadata)
            parent_metadata.update({
                "parent_id": parent_id,
                "parent_index": parent_index,
                "index_schema_version": INDEX_SCHEMA_VERSION,
            })
            normalized_parent = Document(
                page_content=parent.page_content, metadata=parent_metadata
            )
            parents.append(normalized_parent)
            child_parts = split_structured_document(
                normalized_parent,
                chunk_size=child_size,
                chunk_overlap=child_overlap,
            )
            for child_index, child in enumerate(child_parts):
                child_id = _stable_id("CHD", parent_id, child_index, child.page_content)
                child_metadata = dict(child.metadata)
                child_metadata.update({
                    "child_id": child_id,
                    "child_index": child_index,
                    "parent_id": parent_id,
                    "parent_chunk_index": parent_index,
                })
                children.append(Document(
                    page_content=child.page_content, metadata=child_metadata
                ))
    return parents, children


def _document_digest(documents: list[Document]) -> str:
    hasher = hashlib.sha256()
    for document in documents:
        hasher.update(_source_key(document).encode("utf-8", errors="ignore"))
        hasher.update(b"\0")
        hasher.update(document.page_content.encode("utf-8", errors="ignore"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def _write_documents(path: Path, documents: list[Document]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for document in documents:
            handle.write(json.dumps({
                "page_content": document.page_content,
                "metadata": _json_safe_metadata(document.metadata),
            }, ensure_ascii=False) + "\n")


def _read_documents(path: Path) -> list[Document]:
    documents: list[Document] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or not isinstance(value.get("metadata"), dict):
                raise RuntimeError(f"Invalid RAG document at {path}:{line_number}")
            documents.append(Document(
                page_content=str(value.get("page_content") or ""),
                metadata=value["metadata"],
            ))
    return documents


def _index_parameters() -> dict[str, int]:
    return {
        "parent_chunk_size": get_rag_parent_chunk_size(),
        "parent_chunk_overlap": get_rag_parent_chunk_overlap(),
        "child_chunk_size": get_rag_child_chunk_size(),
        "child_chunk_overlap": get_rag_child_chunk_overlap(),
    }


def _pdf_parameters() -> dict[str, Any]:
    """Return parsing settings that affect PDF text or index provenance."""
    return {
        "extract_tables": get_pdf_extract_tables(),
        "min_text_chars": get_pdf_min_text_chars(),
        "max_pages": get_pdf_max_pages(),
        "max_file_mb": get_pdf_max_file_mb(),
        "ocr_mode": get_pdf_ocr_mode(),
        "ocr_language": get_pdf_ocr_language(),
    }


def _load_manifest() -> dict[str, Any] | None:
    if not INDEX_MANIFEST_FILE.exists():
        return None
    try:
        value = json.loads(INDEX_MANIFEST_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def academic_section_prior(section_title: str, query: str) -> float:
    """Apply a small, explicit paper-section prior without hiding references."""
    section = str(section_title or "").strip().lower()
    lowered_query = str(query or "").lower()
    asks_for_references = any(marker in lowered_query for marker in (
        "reference", "bibliography", "citation", "参考文献", "引用文献"
    ))
    if any(marker in section for marker in ("reference", "bibliography", "参考文献")):
        return 1.0 if asks_for_references else 0.65
    if any(marker in section for marker in (
        "abstract", "result", "discussion", "conclusion",
        "摘要", "结果", "讨论", "结论",
    )):
        return 1.08
    return 1.0


def build_knowledge_index(*, force: bool = False) -> dict[str, Any]:
    documents = load_documents()
    if not documents:
        raise RuntimeError(f"No supported documents found in {KNOWLEDGE_DIR}")

    parameters = _index_parameters()
    pdf_parameters = _pdf_parameters()
    corpus_digest = _document_digest(documents)
    embedding_model = get_embedding_model_path()
    manifest = _load_manifest()
    required_files = (
        INDEX_DIR / "index.faiss",
        INDEX_DIR / "index.pkl",
        PARENT_STORE_FILE,
        CHILD_STORE_FILE,
    )
    if (
        not force
        and manifest
        and manifest.get("schema_version") == INDEX_SCHEMA_VERSION
        and manifest.get("corpus_digest") == corpus_digest
        and manifest.get("embedding_model") == embedding_model
        and manifest.get("parameters") == parameters
        and manifest.get("pdf_parser") == PARSER_NAME
        and manifest.get("pdf_parameters") == pdf_parameters
        and all(path.exists() for path in required_files)
    ):
        return {
            "status": "unchanged",
            "source_documents": int(manifest.get("source_documents", len(documents))),
            "parent_chunks": int(manifest.get("parent_chunks", 0)),
            "child_chunks": int(manifest.get("child_chunks", 0)),
            "corpus_digest": corpus_digest,
        }

    parents, children = create_parent_child_chunks(documents, **parameters)
    if not children:
        raise RuntimeError("Knowledge documents produced no child chunks")
    vector_store = FAISS.from_documents(children, create_embeddings())
    new_manifest = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": embedding_model,
        "corpus_digest": corpus_digest,
        "source_documents": len(documents),
        "parent_chunks": len(parents),
        "child_chunks": len(children),
        "parameters": parameters,
        "retrieval": "dense+bm25+rrf",
        "pdf_parser": PARSER_NAME,
        "pdf_parameters": pdf_parameters,
        "reranker_model": get_reranker_model_path(),
    }

    INDEX_DIR.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="rag-index-", dir=str(INDEX_DIR.parent)
    ) as temporary:
        temporary_path = Path(temporary)
        vector_store.save_local(str(temporary_path))
        _write_documents(temporary_path / PARENT_STORE_FILE.name, parents)
        _write_documents(temporary_path / CHILD_STORE_FILE.name, children)
        (temporary_path / INDEX_MODEL_FILE.name).write_text(
            embedding_model, encoding="utf-8"
        )
        (temporary_path / INDEX_MANIFEST_FILE.name).write_text(
            json.dumps(new_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        # Manifest is replaced last. A failed build therefore cannot advertise
        # a partially written index as a complete new version.
        for name in (
            "index.faiss", "index.pkl", PARENT_STORE_FILE.name,
            CHILD_STORE_FILE.name, INDEX_MODEL_FILE.name,
        ):
            os.replace(temporary_path / name, INDEX_DIR / name)
        os.replace(temporary_path / INDEX_MANIFEST_FILE.name, INDEX_MANIFEST_FILE)

    return {
        "status": "rebuilt",
        "source_documents": len(documents),
        "parent_chunks": len(parents),
        "child_chunks": len(children),
        "corpus_digest": corpus_digest,
    }


class KnowledgeBase:
    def __init__(self) -> None:
        manifest = _load_manifest()
        if not manifest or manifest.get("schema_version") != INDEX_SCHEMA_VERSION:
            raise FileNotFoundError(
                "Parent-child knowledge index is missing or outdated. "
                "Run `python -m scripts.build_index --force`."
            )
        embedding_model_path = get_embedding_model_path()
        if manifest.get("embedding_model") != embedding_model_path:
            raise RuntimeError(
                "Knowledge index was not built with the configured embedding "
                f"model ({embedding_model_path}). Rebuild the index."
            )
        for path in (
            INDEX_DIR / "index.faiss", INDEX_DIR / "index.pkl",
            PARENT_STORE_FILE, CHILD_STORE_FILE,
        ):
            if not path.exists():
                raise FileNotFoundError(f"Knowledge index file is missing: {path.name}")

        self.manifest = manifest
        self.embeddings = create_embeddings()
        self.vector_store = FAISS.load_local(
            str(INDEX_DIR),
            self.embeddings,
            # The pickle is generated by this project's build command. Never
            # replace data/index with files from an untrusted source.
            allow_dangerous_deserialization=True,
        )
        parents = _read_documents(PARENT_STORE_FILE)
        self.parents = {
            str(document.metadata["parent_id"]): document for document in parents
        }
        self.children = _read_documents(CHILD_STORE_FILE)
        self.children_by_id = {
            str(document.metadata["child_id"]): document for document in self.children
        }
        self.lexical_index = BM25Index(
            self.children_by_id.keys(),
            (document.page_content for document in self.children_by_id.values()),
        )
        self.reranker_model_path = get_reranker_model_path()
        self._reranker: LocalCrossEncoderReranker | None = None
        self._reranker_failed = False

    def _rerank(
        self, query: str, child_ids: list[str]
    ) -> tuple[list[str], dict[str, float], str]:
        if not self.reranker_model_path or self._reranker_failed:
            status = "not_configured" if not self.reranker_model_path else "failed"
            return child_ids, {}, status
        try:
            if self._reranker is None:
                self._reranker = LocalCrossEncoderReranker(self.reranker_model_path)
            scores = self._reranker.score(
                query,
                [self.children_by_id[item_id].page_content for item_id in child_ids],
            )
            score_map = dict(zip(child_ids, scores))
            return (
                sorted(child_ids, key=lambda item_id: (-score_map[item_id], item_id)),
                score_map,
                "applied",
            )
        except Exception:
            self._reranker_failed = True
            logger.exception("local_rag_reranker_failed")
            return child_ids, {}, "failed"

    def search(self, query: str, top_k: int = 4) -> list[Document]:
        top_k = max(1, min(int(top_k), 12))
        fetch_k = max(top_k * 6, 24)
        dense_documents = self.vector_store.similarity_search(query, k=fetch_k)
        dense_ids = [
            str(document.metadata.get("child_id") or "")
            for document in dense_documents
            if document.metadata.get("child_id")
        ]
        lexical_hits = self.lexical_index.search(query, fetch_k)
        lexical_ids = [hit.item_id for hit in lexical_hits]
        fused_scores = reciprocal_rank_fusion(
            {"dense": dense_ids, "bm25": lexical_ids},
            weights={"dense": 1.0, "bm25": 1.0},
            rank_constant=get_rag_rrf_k(),
        )
        section_priors = {
            item_id: academic_section_prior(
                self.children_by_id[item_id].metadata.get("section_title", ""),
                query,
            )
            for item_id in fused_scores
            if item_id in self.children_by_id
        }
        candidate_ids = sorted(
            fused_scores,
            key=lambda item_id: (
                -(fused_scores[item_id] * section_priors.get(item_id, 1.0)),
                item_id,
            ),
        )[:fetch_k]
        candidate_ids, rerank_scores, reranker_status = self._rerank(
            query, candidate_ids
        )
        dense_ranks = {item_id: rank for rank, item_id in enumerate(dense_ids, 1)}
        lexical_ranks = {
            item_id: rank for rank, item_id in enumerate(lexical_ids, 1)
        }

        selected: dict[str, Document] = {}
        matched_children: dict[str, list[str]] = {}
        for child_id in candidate_ids:
            child = self.children_by_id.get(child_id)
            if child is None:
                continue
            parent_id = str(child.metadata.get("parent_id") or "")
            parent = self.parents.get(parent_id)
            if parent is None:
                continue
            matched_children.setdefault(parent_id, []).append(child_id)
            if parent_id in selected or len(selected) >= top_k:
                continue
            metadata = dict(parent.metadata)
            metadata.update({
                "matched_child_id": child_id,
                "dense_rank": dense_ranks.get(child_id),
                "lexical_rank": lexical_ranks.get(child_id),
                "rrf_score": round(float(fused_scores.get(child_id, 0.0)), 8),
                "section_prior": section_priors.get(child_id, 1.0),
                "rerank_score": rerank_scores.get(child_id),
                "reranker_status": reranker_status,
                "retrieval_strategy": (
                    "parent_child_dense_bm25_rrf_rerank"
                    if reranker_status == "applied"
                    else "parent_child_dense_bm25_rrf"
                ),
            })
            selected[parent_id] = Document(
                page_content=parent.page_content, metadata=metadata
            )

        results = list(selected.values())
        for document in results:
            parent_id = str(document.metadata.get("parent_id") or "")
            document.metadata["matched_child_ids"] = matched_children.get(
                parent_id, []
            )[:5]
        return results


_knowledge_base: KnowledgeBase | None = None


def _get_knowledge_base() -> KnowledgeBase:
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = KnowledgeBase()
    return _knowledge_base


def retrieve_knowledge(query: str, top_k: int = 4) -> list[dict[str, Any]]:
    documents = _get_knowledge_base().search(query, top_k)
    results: list[dict[str, Any]] = []
    for doc in documents:
        source = doc.metadata.get("file_name")
        if not source:
            source = Path(str(doc.metadata.get("source", "Unknown"))).name
        results.append({
            "title": source,
            "content": doc.page_content,
            "file_path": doc.metadata.get("file_path", ""),
            "page": doc.metadata.get("page"),
            "page_start": doc.metadata.get("page_start"),
            "page_end": doc.metadata.get("page_end"),
            "section_title": doc.metadata.get("section_title", ""),
            "content_types": doc.metadata.get("content_types", ""),
            "parser": doc.metadata.get("parser", ""),
            "parser_warnings": doc.metadata.get("parser_warnings", ""),
            "ocr_used": doc.metadata.get("ocr_used", False),
            "doi": doc.metadata.get("doi", ""),
            "parent_id": doc.metadata.get("parent_id", ""),
            "matched_child_ids": doc.metadata.get("matched_child_ids", []),
            "retrieval_strategy": doc.metadata.get("retrieval_strategy", ""),
            "rrf_score": doc.metadata.get("rrf_score"),
            "section_prior": doc.metadata.get("section_prior"),
            "rerank_score": doc.metadata.get("rerank_score"),
            "reranker_status": doc.metadata.get("reranker_status"),
        })
    return results


def search_knowledge(query: str, top_k: int = 4) -> str:
    results = retrieve_knowledge(query, top_k)
    if not results:
        return "No relevant local knowledge found."
    texts = []
    for index, item in enumerate(results, 1):
        location = str(item["title"])
        if item.get("page_start"):
            location += f", page {item['page_start']}"
        elif item["page"] is not None:
            location += f", page {int(item['page']) + 1}"
        if item.get("section_title"):
            location += f", section {item['section_title']}"
        texts.append(f"[Local {index}] {location}\n{item['content']}")
    return "\n\n".join(texts)
