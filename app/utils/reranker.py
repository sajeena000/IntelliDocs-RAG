from __future__ import annotations
from typing import List, Dict, Optional
from threading import Lock
from sentence_transformers import CrossEncoder

from app.core.config import get_settings

_settings = get_settings()
_singleton_lock = Lock()


class Reranker:
    _instance: Optional["Reranker"] = None

    def __init__(self, model_path: str):
        self._model = CrossEncoder(model_path)

    @classmethod
    def get(cls) -> "Reranker":
        if cls._instance is None:
            with _singleton_lock:
                if cls._instance is None:
                    cls._instance = Reranker(_settings.RERANKER_MODEL_PATH)
        return cls._instance

    def rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        if not candidates:
            return []
        pairs = [(query, c.get("text", "")) for c in candidates]
        scores = self._model.predict(pairs).tolist()
        ranked = sorted(
            [{"score": float(s), **c} for s, c in zip(scores, candidates)],
            key=lambda x: x["score"],
            reverse=True,
        )
        return ranked