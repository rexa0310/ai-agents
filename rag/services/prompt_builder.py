from rag.schemas import RetrievalResult


def build_grounded_prompt(question: str, retrieved_chunks: list[RetrievalResult]) -> str:
    if not retrieved_chunks:
        context = "No relevant context was retrieved from the knowledge base."
    else:
        context = "\n\n".join(
            [
                (
                    f"[Chunk {item.chunk_id} | Source {item.source_id} | "
                    f"Score {item.score:.4f}]\n{item.text}"
                )
                for item in retrieved_chunks
            ]
        )

    return f"""
You are a helpful RAG assistant.

Answer the user's question using the provided context.
If the answer is not grounded in the context, say that the knowledge base does not contain enough information.
Do not invent citations or facts.

Question:
{question}

Context:
{context}
""".strip()
