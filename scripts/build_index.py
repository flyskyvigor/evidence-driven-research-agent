from research_agent.rag.knowledge_base import build_knowledge_index


def main():
    document_count, chunk_count = build_knowledge_index()

    print("Knowledge index built successfully.")
    print(f"Documents: {document_count}")
    print(f"Chunks: {chunk_count}")


if __name__ == "__main__":
    main()
