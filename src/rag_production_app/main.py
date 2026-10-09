import asyncio
import logging
import uuid

import inngest
import inngest.fast_api
from fastapi import FastAPI
from inngest.experimental import ai

from rag_production_app.config import settings
from rag_production_app.custom_types import (
    RAGChunkAndSrc,
    RAGSearchResult,
    RAGUpsertResult,
)
from rag_production_app.data_loader import embed_texts, load_and_chunk_pdf
from rag_production_app.vector_db import QdrantStorage

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)


@inngest_client.create_function(
    fn_id="RAG: Inngest PDF", trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_ingest_pdf(ctx: inngest.Context):
    async def _load() -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = await asyncio.to_thread(load_and_chunk_pdf, pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    async def _upsert(data: RAGChunkAndSrc) -> RAGUpsertResult:
        vecs = await embed_texts(
            [c.text for c in data.chunks], prefix="search_document: "
        )
        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{data.source_id}:{i}"))
            for i in range(len(data.chunks))
        ]
        payloads = [
            {"source": data.source_id, "page": c.page, "text": c.text}
            for c in data.chunks
        ]
        store = QdrantStorage()
        store.delete_source(data.source_id)
        store.upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(data.chunks))

    data = await ctx.step.run("load-and-chunk", _load, output_type=RAGChunkAndSrc)
    result = await ctx.step.run(
        "embed-and-upsert", lambda: _upsert(data), output_type=RAGUpsertResult
    )
    return result.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query PDF", trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))
    source = ctx.event.data.get("source")  # optional: restrict to one document

    async def _search() -> RAGSearchResult:
        qvec = (await embed_texts([question], prefix="search_query: "))[0]
        hits = QdrantStorage().search(qvec, top_k, source=source)
        return RAGSearchResult(hits=hits)

    found = await ctx.step.run("embed-and-search", _search, output_type=RAGSearchResult)

    if not found.hits:
        return {
            "answer": "I couldn't find anything relevant in the uploaded documents.",
            "sources": [],
            "num_contexts": 0,
        }

    context_block = "\n\n".join(
        f"[{i}] ({h.source}, p.{h.page})\n{h.text}" for i, h in enumerate(found.hits, 1)
    )
    adapter = ai.openai.Adapter(
        auth_key="ollama", model=settings.llm_model, base_url=settings.ollama_base_url
    )
    res = await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            "max_tokens": 1024,
            "temperature": 0.2,
            "reasoning_effort": "none",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer using only the numbered context. Cite the passages "
                        "you use like [1]. If the context does not contain the "
                        "answer, say you don't know."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context_block}\n\nQuestion: {question}",
                },
            ],
        },
    )
    return {
        "answer": res["choices"][0]["message"]["content"].strip(),
        "sources": sorted({f"{h.source}, p.{h.page}" for h in found.hits}),
        "num_contexts": len(found.hits),
    }


app = FastAPI()

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])
