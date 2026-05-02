from dataclasses import dataclass
from django.conf import settings
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from .ollama_client import embed_text

@dataclass
class RetrievedChunk:
    text: str
    source: str
    tag: str
    score: float

_client = None

def qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=settings.QDRANT_URL)
    return _client

def ensure_collection(vector_size: int):
    client = qdrant_client()
    name = settings.QDRANT_COLLECTION
    existing = [c.name for c in client.get_collections().collections]
    if name in existing:
        return

    client.create_collection(
        collection_name=name,
        vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
    )

def upsert_chunks(chunks: list[dict]):
    """
    chunks: [{id, text, source, tag}]
    """
    client = qdrant_client()
    # embed first chunk to get vector size
    vec0 = embed_text(chunks[0]["text"])
    ensure_collection(vector_size=len(vec0))

    points = []
    for c in chunks:
        v = embed_text(c["text"])
        points.append(
            qm.PointStruct(
                id=c["id"],
                vector=v,
                payload={"text": c["text"], "source": c.get("source", ""), "tag": c.get("tag", "")},
            )
        )

    client.upsert(collection_name=settings.QDRANT_COLLECTION, points=points)

def retrieve_context(query: str) -> list[RetrievedChunk]:
    client = qdrant_client()
    qv = embed_text(query)

    hits = client.search(
        collection_name=settings.QDRANT_COLLECTION,
        query_vector=qv,
        limit=settings.RAG_TOP_K,
        with_payload=True,
    )

    out: list[RetrievedChunk] = []
    for h in hits:
        p = h.payload or {}
        out.append(
            RetrievedChunk(
                text=str(p.get("text", "")),
                source=str(p.get("source", "")),
                tag=str(p.get("tag", "")),
                score=float(h.score),
            )
        )
    return out
