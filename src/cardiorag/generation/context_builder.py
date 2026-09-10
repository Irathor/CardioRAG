"""Builds the numbered [Source N] context block fed to the LLM.

Each block keeps a passage's text tightly bound to the metadata a citation
needs (title, page, DOI). The LLM sees exactly this text under this
numbering, so the app can reconstruct the same source list afterward for
the UI/citations by index, rather than trying to re-derive which chunk the
model meant from its prose output.
"""

from cardiorag.models import RetrievedChunk


def build_context(retrieved_chunks: list[RetrievedChunk]) -> str:
    if not retrieved_chunks:
        return ""

    blocks = []
    for i, retrieved in enumerate(retrieved_chunks, start=1):
        chunk = retrieved.chunk
        pages = ", ".join(str(p) for p in chunk.page_numbers) if chunk.page_numbers else "Unknown"
        blocks.append(
            f"[Source {i}]\n"
            f"Title: {chunk.title or 'Unknown'}\n"
            f"Page(s): {pages}\n"
            f"DOI: {chunk.doi or 'N/A'}\n"
            f"Text: {chunk.text}"
        )
    return "\n\n".join(blocks)
