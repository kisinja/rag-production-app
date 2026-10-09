from openai import AsyncOpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter

from rag_production_app.config import settings
from rag_production_app.custom_types import Chunk

client = AsyncOpenAI(base_url=settings.ollama_base_url, api_key="ollama")
splitter = SentenceSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
)


def load_and_chunk_pdf(path: str) -> list[Chunk]:
    """Sync and CPU-bound. Call it with asyncio.to_thread."""
    chunks: list[Chunk] = []
    for doc in PDFReader().load_data(file=path):
        if not doc.text:
            continue
        page = doc.metadata.get("page_label")
        for text in splitter.split_text(doc.text):
            chunks.append(Chunk(text=text, page=page))

    return chunks


async def embed_texts(texts: list[str], prefix: str = "") -> list[list[float]]:
    vectors: list[list[float]] = []
    size = settings.embed_batch_size
    for i in range(0, len(texts), size):
        batch = [prefix + t for t in texts[i : i + size]]
        resp = await client.embeddings.create(model=settings.embed_model, input=batch)
        vectors.extend(item.embedding for item in resp.data)
    return vectors
