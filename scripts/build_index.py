import argparse

from research_agent.rag.knowledge_base import build_knowledge_index


def _arguments():
    parser = argparse.ArgumentParser(description="Build the local parent-child RAG index")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even when source digest and index settings are unchanged",
    )
    return parser.parse_args()


def main():
    args = _arguments()
    result = build_knowledge_index(force=args.force)

    print(f"Knowledge index status: {result['status']}")
    print(f"Source documents/pages: {result['source_documents']}")
    print(f"Parent chunks: {result['parent_chunks']}")
    print(f"Child chunks: {result['child_chunks']}")
    print(f"Corpus digest: {result['corpus_digest']}")


if __name__ == "__main__":
    main()
