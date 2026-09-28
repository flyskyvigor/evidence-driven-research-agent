"""Optional local CrossEncoder reranker.

The reranker is deliberately optional because it adds model memory and latency.
When no local path is configured the retrieval pipeline remains a real hybrid
retriever (dense + BM25 + RRF), but must not be described as reranked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


class LocalCrossEncoderReranker:
    def __init__(self, model_path: str) -> None:
        if not model_path:
            raise ValueError("model_path is required")
        local_path = Path(model_path).expanduser()
        if not local_path.exists() or not local_path.is_dir():
            raise FileNotFoundError(
                "RERANKER_MODEL_PATH must point to an existing local model directory"
            )
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(str(local_path), device="cpu")

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        values = self.model.predict(
            [(query, passage) for passage in passages],
            show_progress_bar=False,
        )
        return [float(value) for value in values]

