import asyncio
import json
import sys
import uuid
from pathlib import Path

from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from rag_production_app.config import settings
from rag_production_app.data_loader import embed_texts

GRID = [(300, 0), (300, 50), (500, 50), (500, 100), (1000, 0), (1000, 100)]


async def build_collection(qc, name, pdfs, chunk_size, overlap) -> int:
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    if qc.collection_exists(name):
        qc.delete_collection(name)
    qc.create_collection(
        name,
        vectors_config=VectorParams(size=settings.embed_dim, distance=Distance.COSINE),
    )
    total = 0
    for pdf in pdfs:
        source = Path(pdf).name
        texts, pages = [], []
        for doc in PDFReader().load_data(file=pdf):
            for t in splitter.split_text(doc.text or ""):
                texts.append(t)
                pages.append(doc.metadata.get("page_label"))
        vecs = await embed_texts(texts, prefix="search_document: ")
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{name}:{source}:{total + i}")),
                vector=v,
                payload={"source": source, "page": p, "text": t},
            )
            for i, (t, p, v) in enumerate(zip(texts, pages, vecs))
        ]
        qc.upsert(name, points=points)
        total += len(points)
    return total


async def evaluate(qc, name, rows, k) -> tuple[float, float]:
    qvecs = await embed_texts([r["question"] for r in rows], prefix="search_query: ")
    hits, rr = 0, 0.0
    for r, qv in zip(rows, qvecs):
        found = qc.query_points(name, query=qv, limit=k, with_payload=True).points
        rank = next(
            (
                i
                for i, p in enumerate(found, 1)
                if p.payload["source"] == r["source"] and p.payload["page"] == r["page"]
            ),
            None,
        )
        if rank:
            hits += 1
            rr += 1 / rank
    return hits / len(rows), rr / len(rows)


async def main(dataset: str, pdfs: list[str], k: int = 5) -> None:
    rows = [
        json.loads(l)
        for l in Path(dataset).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    qc = QdrantClient(url=settings.qdrant_url)
    print(f"{len(rows)} questions, top-{k}, no score threshold\n")
    print(f"| chunk size | overlap | chunks | hit@{k} | MRR |")
    print("|---|---|---|---|---|")
    for cs, ov in GRID:
        name = f"eval_{cs}_{ov}"
        n = await build_collection(qc, name, pdfs, cs, ov)
        hit, mrr = await evaluate(qc, name, rows, k)
        print(f"| {cs} | {ov} | {n} | {hit:.2f} | {mrr:.3f} |")
        qc.delete_collection(name)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2:]))
