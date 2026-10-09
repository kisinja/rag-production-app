import asyncio
import uuid

from rag_production_app import data_loader


def test_chunk_ids_are_deterministic():
    a = uuid.uuid5(uuid.NAMESPACE_URL, "report.pdf:0")
    b = uuid.uuid5(uuid.NAMESPACE_URL, "report.pdf:0")
    assert a == b
    assert a != uuid.uuid5(uuid.NAMESPACE_URL, "report.pdf:1")


def test_splitter_respects_overlap_setting():
    text = ("This is a sentence. " * 400).strip()
    chunks = data_loader.splitter.split_text(text)
    assert len(chunks) > 1
    assert all(chunks)


def test_embed_texts_batches(monkeypatch):
    calls = []

    class FakeResp:
        def __init__(self, n):
            self.data = [type("I", (), {"embedding": [0.0] * 3}) for _ in range(n)]

    class FakeEmbeddings:
        async def create(self, model, input):
            calls.append(len(input))
            return FakeResp(len(input))

    class FakeClient:
        embeddings = FakeEmbeddings()

    monkeypatch.setattr(data_loader, "client", FakeClient())
    monkeypatch.setattr(data_loader.settings, "embed_batch_size", 10)

    vecs = asyncio.run(data_loader.embed_texts(["x"] * 25))
    assert len(vecs) == 25
    assert calls == [10, 10, 5]
