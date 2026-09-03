import numpy as np
from sentence_transformers import SentenceTransformer

from research_agent.config import get_embedding_model_path

_model = None


def _get_model():
    global _model

    if _model is None:
        _model = SentenceTransformer(
            get_embedding_model_path(),
            device="cpu"
        )

    return _model


def semantic_relevance_scores(question, evidence):
    if not evidence:
        return []

    # BGE-M3 directly embeds multilingual questions and evidence in one space.
    query = question

    passages = []

    for item in evidence:
        title = item.get("title", "")
        content = item.get("content", "")[:1800]

        passages.append(
            f"{title}\n{content}"
        )

    model = _get_model()

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    )[0]

    passage_embeddings = model.encode(
        passages,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    similarities = (
        passage_embeddings @ query_embedding
    )

    return [
        float(np.clip(score, 0.0, 1.0))
        for score in similarities
    ]
