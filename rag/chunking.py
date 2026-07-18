def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    cleaned_text = " ".join(text.split())
    if not cleaned_text:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    step = chunk_size - chunk_overlap

    while start < len(cleaned_text):
        end = start + chunk_size
        chunks.append(cleaned_text[start:end])
        start += step

    return chunks
