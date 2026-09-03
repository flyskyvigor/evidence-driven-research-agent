from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from research_agent.config import get_embedding_model_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_DIR = PROJECT_ROOT / "data" / "knowledge"
INDEX_DIR = PROJECT_ROOT / "data" / "index"
INDEX_MODEL_FILE = INDEX_DIR / "embedding_model.txt"


def create_embeddings():
    return HuggingFaceEmbeddings(
        model_name=get_embedding_model_path(),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def load_documents():
    documents = []

    for path in sorted(KNOWLEDGE_DIR.rglob("*")):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()

        try:
            if suffix == ".pdf":
                docs = PyPDFLoader(str(path)).load()
            elif suffix in {".txt", ".md"}:
                docs = TextLoader(
                    str(path),
                    encoding="utf-8",
                    autodetect_encoding=True
                ).load()
            else:
                continue
        except Exception as e:
            print(f"Skip {path.name}: {e}")
            continue

        for doc in docs:
            doc.metadata["file_name"] = path.name
            doc.metadata["file_path"] = str(path)

        documents.extend(docs)

    return documents


def build_knowledge_index():
    documents = load_documents()

    if not documents:
        raise RuntimeError(
            f"No supported documents found in {KNOWLEDGE_DIR}"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    embeddings = create_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(INDEX_DIR))
    # A FAISS index must be rebuilt whenever the embedding model changes.
    INDEX_MODEL_FILE.write_text(
        get_embedding_model_path(),
        encoding="utf-8"
    )

    return len(documents), len(chunks)


class KnowledgeBase:
    def __init__(self):
        if not (INDEX_DIR / "index.faiss").exists():
            raise FileNotFoundError(
                "Knowledge index does not exist. "
                "Run `python -m scripts.build_index` first."
            )

        index_model = (
            INDEX_MODEL_FILE.read_text(encoding="utf-8").strip()
            if INDEX_MODEL_FILE.exists()
            else None
        )
        embedding_model_path = get_embedding_model_path()
        if index_model != embedding_model_path:
            raise RuntimeError(
                "Knowledge index was not built with the configured embedding "
                f"model ({embedding_model_path}). "
                "Run `python -m scripts.build_index` to rebuild it."
            )

        self.embeddings = create_embeddings()
        self.vector_store = FAISS.load_local(
            str(INDEX_DIR),
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    def search(self, query, top_k=4):
        retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": top_k,
                "fetch_k": max(top_k * 3, 12),
                "lambda_mult": 0.7
            }
        )

        return retriever.invoke(query)


_knowledge_base = None


def _get_knowledge_base():
    global _knowledge_base

    if _knowledge_base is None:
        _knowledge_base = KnowledgeBase()

    return _knowledge_base


def retrieve_knowledge(query, top_k=4):
    documents = _get_knowledge_base().search(query, top_k)
    results = []

    for doc in documents:
        source = doc.metadata.get("file_name")
        if not source:
            source = Path(doc.metadata.get("source", "Unknown")).name

        results.append({
            "title": source,
            "content": doc.page_content,
            "file_path": doc.metadata.get("file_path", ""),
            "page": doc.metadata.get("page")
        })

    return results


def search_knowledge(query, top_k=4):
    results = retrieve_knowledge(query, top_k)
    if not results:
        return "No relevant local knowledge found."

    texts = []
    for i, item in enumerate(results, 1):
        location = item["title"]
        if item["page"] is not None:
            location += f", page {item['page'] + 1}"
        texts.append(f"[Local {i}] {location}\n{item['content']}")

    return "\n\n".join(texts)
