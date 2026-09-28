"""Deterministic lexical retrieval and rank fusion for the local RAG index.

The vector index remains responsible for semantic recall.  This module adds a
small in-process BM25 implementation so the project can also recover exact
identifiers, API names and uncommon terminology without introducing a second
search service.  Scores from different retrievers are not compared directly;
Reciprocal Rank Fusion combines their ranks instead.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable


_LATIN_TOKEN = re.compile(r"[a-z0-9_./:+#-]+", re.IGNORECASE)
_CJK_RUN = re.compile(r"[\u3400-\u9fff]+")


def tokenize_for_search(text: str) -> list[str]:
    """Tokenize mixed Chinese/English text without an online tokenizer.

    English/code identifiers are kept as words. Chinese runs contribute both
    single characters and bi-grams, which is intentionally simple but makes
    BM25 useful for exact Chinese phrases while BGE handles semantics.
    """
    normalized = str(text or "").lower()
    tokens = _LATIN_TOKEN.findall(normalized)
    for run in _CJK_RUN.findall(normalized):
        tokens.extend(run)
        tokens.extend(run[index:index + 2] for index in range(len(run) - 1))
    return [token for token in tokens if token.strip()]


@dataclass(frozen=True)
class RankedHit:
    item_id: str
    score: float


class BM25Index:
    """A compact BM25 index for the local, single-process knowledge base."""

    def __init__(
        self,
        item_ids: Iterable[str],
        documents: Iterable[str],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.item_ids = list(item_ids)
        self.token_counts = [Counter(tokenize_for_search(text)) for text in documents]
        if len(self.item_ids) != len(self.token_counts):
            raise ValueError("item_ids and documents must have the same length")
        self.k1 = float(k1)
        self.b = float(b)
        self.lengths = [sum(counts.values()) for counts in self.token_counts]
        self.average_length = (
            sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        )
        document_frequency: Counter[str] = Counter()
        for counts in self.token_counts:
            document_frequency.update(counts.keys())
        total = len(self.token_counts)
        self.idf = {
            token: math.log(1.0 + (total - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

    def search(self, query: str, top_k: int) -> list[RankedHit]:
        query_tokens = tokenize_for_search(query)
        if not query_tokens or not self.token_counts:
            return []
        scores: list[RankedHit] = []
        average_length = self.average_length or 1.0
        for item_id, counts, length in zip(
            self.item_ids, self.token_counts, self.lengths
        ):
            score = 0.0
            for token in query_tokens:
                frequency = counts.get(token, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * length / average_length
                )
                score += self.idf.get(token, 0.0) * (
                    frequency * (self.k1 + 1.0) / denominator
                )
            if score > 0:
                scores.append(RankedHit(item_id=item_id, score=score))
        scores.sort(key=lambda item: (-item.score, item.item_id))
        return scores[:max(1, int(top_k))]


def reciprocal_rank_fusion(
    rankings: dict[str, list[str]],
    *,
    weights: dict[str, float] | None = None,
    rank_constant: int = 60,
) -> dict[str, float]:
    """Fuse independent ranked lists without assuming comparable raw scores."""
    if rank_constant < 1:
        raise ValueError("rank_constant must be positive")
    weights = weights or {}
    fused: dict[str, float] = {}
    for name, ranking in rankings.items():
        weight = float(weights.get(name, 1.0))
        seen: set[str] = set()
        for rank, item_id in enumerate(ranking, 1):
            if not item_id or item_id in seen:
                continue
            seen.add(item_id)
            fused[item_id] = fused.get(item_id, 0.0) + weight / (
                rank_constant + rank
            )
    return fused

