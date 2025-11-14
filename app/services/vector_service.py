from __future__ import annotations
from typing import List, Dict, Any, Optional
from threading import Lock

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings

_settings = get_settings()
_singleton_lock = Lock()


class VectorService:
    _instance: Optional["VectorService"] = None

    def __init__(self, client: QdrantClient, collection: str, embedding_model: SentenceTransformer):
        self.client = client
        self.collection = collection
        self.embedding_model = embedding_model
        self._ensure_collection()

    @classmethod
    def get(cls) -> "VectorService":
        if cls._instance is None:
            with _singleton_lock:
                if cls._instance is None:
                    client = QdrantClient(host=_settings.QDRANT_HOST, port=_settings.QDRANT_PORT, timeout=60)
                    model = SentenceTransformer(_settings.EMBEDDING_MODEL_PATH)
                    cls._instance = VectorService(client, _settings.QDRANT_COLLECTION, model)
        return cls._instance

    @staticmethod
    def get_embeddings(texts: List[str]) -> List[List[float]]:
        model = VectorService.get().embedding_model
        return model.encode(texts, normalize_embeddings=True).tolist()

    def _ensure_collection(self) -> None:
        dim = self.embedding_model.get_sentence_embedding_dimension()
        existing = None
        try:
            existing = self.client.get_collection(self.collection)
        except Exception:
            existing = None

        if existing is None:
            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
            )

    def upsert_points(self, ids: List[str], vectors: List[List[float]], payloads: List[Dict[str, Any]]) -> None:
        points = [qmodels.PointStruct(id=pid, vector=vec, payload=pl) for pid, vec, pl in zip(ids, vectors, payloads)]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query_vector: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        results = self.client.search(collection_name=self.collection, query_vector=query_vector, limit=limit)
        out: List[Dict[str, Any]] = []
        for r in results:
            payload = r.payload or {}
            out.append(
                {
                    "score": float(r.score),
                    "document_id": payload.get("document_id"),
                    "chunk_id": payload.get("chunk_id"),
                    "chunk_index": payload.get("chunk_index"),
                    "text": payload.get("text", ""),
                }
            )
        return out