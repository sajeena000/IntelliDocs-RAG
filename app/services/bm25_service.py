from __future__ import annotations
from typing import List, Dict, Optional
from threading import Lock

from rank_bm25 import BM25Okapi
from sqlalchemy.orm import Session

from app.models.models import Chunk

import re

_singleton_lock = Lock()


class BM25Service:
    _instance: Optional["BM25Service"] = None

    def __init__(self):
        self._bm25: Optional[BM25Okapi] = None
        self._chunk_ids: List[str] = []
        self._tokenized_corpus: List[List[str]] = []

    @classmethod
    def get(cls) -> "BM25Service":
        if cls._instance is None:
            with _singleton_lock:
                if cls._instance is None:
                    cls._instance = BM25Service()
        return cls._instance

    def rebuild(self, db: Session) -> None:
        chunks: List[Chunk] = db.query(Chunk).all()
        self._chunk_ids = [str(ch.id) for ch in chunks]
        self._tokenized_corpus = [self._tokenize(ch.text) for ch in chunks]
        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)
        else:
            self._bm25 = None

    def search(self, query: str, top_k: int = 20) -> List[Dict]:
        if not self._bm25:
            return []
        query_tokens = self._tokenize(query)
        scores = self._bm25.get_scores(query_tokens)
        # Get top_k indices
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for idx, score in ranked:
            results.append({"chunk_id": self._chunk_ids[idx], "score": float(score)})
        return results

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = text.lower()
        return re.findall(r"\b[a-z0-9]+\b", text)