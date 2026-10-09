from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

from rag_production_app.config import settings
from rag_production_app.custom_types import Hit


def _source_filter(source: str) -> Filter:
    return Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))])


class QdrantStorage:
    def __init__(self):
        self.client = QdrantClient(url=settings.qdrant_url, timeout=30)
        self.collection = settings.qdrant_collection
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=settings.embed_dim, distance=Distance.COSINE
                ),
            )

    def delete_source(self, source: str) -> None:
        """Removes stale chunks when a shorter version of a doc is re-ingested."""
        self.client.delete(
            self.collection,
            points_selector=FilterSelector(filter=_source_filter(source)),
        )

    def upsert(self, ids, vectors, payloads):
        points = [
            PointStruct(id=i, vector=v, payload=p)
            for i, v, p in zip(ids, vectors, payloads)
        ]
        self.client.upsert(self.collection, points=points)

    def search(
        self, query_vector, top_k: int = 5, source: str | None = None
    ) -> list[Hit]:
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=_source_filter(source) if source else None,
            score_threshold=settings.min_score,
            with_payload=True,
            with_vectors=False,
            limit=top_k,
        )
        return [
            Hit(
                text=p.payload.get("text", ""),
                source=p.payload.get("source", ""),
                page=p.payload.get("page"),
                score=p.score,
            )
            for p in response.points
            if p.payload and p.payload.get("text")
        ]
