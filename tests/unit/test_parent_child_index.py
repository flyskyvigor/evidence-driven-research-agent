from langchain_core.documents import Document

from research_agent.rag.hybrid import RankedHit
from research_agent.rag.knowledge_base import (
    KnowledgeBase,
    academic_section_prior,
    create_parent_child_chunks,
)


class FakeVectorStore:
    def __init__(self, documents):
        self.documents = documents

    def similarity_search(self, query, k):
        return self.documents[:k]


class FakeLexicalIndex:
    def __init__(self, hits):
        self.hits = hits

    def search(self, query, top_k):
        return self.hits[:top_k]


def test_parent_child_chunks_are_stable_and_traceable():
    document = Document(
        page_content=(
            "第一章介绍父子索引的背景。" * 20
            + "第二章介绍BM25与稠密检索融合。" * 20
        ),
        metadata={"file_name": "guide.md", "page": 0},
    )
    parents, children = create_parent_child_chunks(
        [document],
        parent_chunk_size=220,
        parent_chunk_overlap=30,
        child_chunk_size=70,
        child_chunk_overlap=10,
    )
    repeated_parents, repeated_children = create_parent_child_chunks(
        [document],
        parent_chunk_size=220,
        parent_chunk_overlap=30,
        child_chunk_size=70,
        child_chunk_overlap=10,
    )

    parent_ids = {item.metadata["parent_id"] for item in parents}
    assert len(parents) > 1
    assert len(children) > len(parents)
    assert all(item.metadata["parent_id"] in parent_ids for item in children)
    assert [item.metadata["parent_id"] for item in parents] == [
        item.metadata["parent_id"] for item in repeated_parents
    ]
    assert [item.metadata["child_id"] for item in children] == [
        item.metadata["child_id"] for item in repeated_children
    ]


def test_search_returns_parent_while_preserving_matched_child_trace():
    child_a = Document(
        page_content="small exact match",
        metadata={"child_id": "CHD-A", "parent_id": "PAR-A"},
    )
    child_b = Document(
        page_content="lexical match",
        metadata={"child_id": "CHD-B", "parent_id": "PAR-A"},
    )
    parent = Document(
        page_content="larger complete context",
        metadata={"parent_id": "PAR-A", "file_name": "guide.md"},
    )
    knowledge_base = KnowledgeBase.__new__(KnowledgeBase)
    knowledge_base.vector_store = FakeVectorStore([child_a])
    knowledge_base.children_by_id = {"CHD-A": child_a, "CHD-B": child_b}
    knowledge_base.parents = {"PAR-A": parent}
    knowledge_base.lexical_index = FakeLexicalIndex([RankedHit("CHD-B", 3.0)])
    knowledge_base.reranker_model_path = None
    knowledge_base._reranker = None
    knowledge_base._reranker_failed = False

    results = knowledge_base.search("match", top_k=1)
    assert results[0].page_content == "larger complete context"
    assert results[0].metadata["parent_id"] == "PAR-A"
    assert set(results[0].metadata["matched_child_ids"]) == {"CHD-A", "CHD-B"}
    assert results[0].metadata["retrieval_strategy"] == "parent_child_dense_bm25_rrf"


def test_reference_section_is_only_downweighted_when_query_does_not_request_it():
    assert academic_section_prior("References", "compare experiment results") < 1.0
    assert academic_section_prior("References", "list cited references") == 1.0
    assert academic_section_prior("Results", "compare experiment results") > 1.0

