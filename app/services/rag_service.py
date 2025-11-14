from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional
from threading import Lock

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.services.vector_service import VectorService
from app.services.bm25_service import BM25Service
from app.utils.reranker import Reranker
from app.utils.llm_clients import LLMRouter

_settings = get_settings()
_singleton_lock = Lock()


@dataclass
class RAGResult:
    reply: str
    sources: List[Dict]


class RAGService:
    _instance: Optional["RAGService"] = None

    def __init__(self):
        self.vector = VectorService.get()
        self.bm25 = BM25Service.get()
        self.reranker = Reranker.get()

    @classmethod
    def get(cls) -> "RAGService":
        if cls._instance is None:
            with _singleton_lock:
                if cls._instance is None:
                    cls._instance = RAGService()
        return cls._instance

    def answer(
        self,
        user_message: str,
        session_id: str,
        history: List[Dict[str, str]],
        model: str,
        db: Session,
    ) -> RAGResult:
        # Dense
        q_vec = self.vector.get_embeddings([user_message])[0]
        dense_hits = self.vector.search(q_vec, limit=_settings.TOP_K_DENSE)

        # BM25 (ensure index exists)
        if self.bm25._bm25 is None:
            self.bm25.rebuild(db)
        bm25_hits = self.bm25.search(user_message, top_k=_settings.TOP_K_BM25)

        # Fetch BM25 hit payloads from DB (chunk_id -> text, doc_id, index)
        chunk_map = {}
        if bm25_hits:
            from app.models.models import Chunk
            chunk_ids = [h["chunk_id"] for h in bm25_hits]
            rows = db.query(Chunk).filter(Chunk.id.in_(chunk_ids)).all()
            for r in rows:
                chunk_map[str(r.id)] = {
                    "document_id": str(r.document_id),
                    "chunk_id": str(r.id),
                    "chunk_index": r.chunk_index,
                    "text": r.text,
                }

        bm25_payloads = []
        for h in bm25_hits:
            meta = chunk_map.get(h["chunk_id"])
            if meta:
                bm25_payloads.append({**meta, "score": h["score"]})

        # Merge
        candidates: Dict[str, Dict] = {}
        for item in dense_hits:
            candidates[item["chunk_id"]] = item
        for item in bm25_payloads:
            if item["chunk_id"] in candidates:
                # combine scores (simple sum after normalization not known; keep max)
                candidates[item["chunk_id"]]["score"] = max(
                    float(candidates[item["chunk_id"]]["score"]), float(item["score"])
                )
            else:
                candidates[item["chunk_id"]] = item

        merged = list(candidates.values())
        # Re-rank with cross-encoder
        reranked = self.reranker.rerank(user_message, merged)
        top = reranked[: _settings.TOP_K_FINAL]

        context = self._build_context(top)
        prompt = self._build_prompt(history, user_message, context)

        llm = LLMRouter.get(model)
        reply = llm.generate(prompt)

        return RAGResult(reply=reply, sources=top)

    def _build_context(self, chunks: List[Dict]) -> str:
        pieces: List[str] = []
        total = 0
        for ch in chunks:
            text = ch.get("text", "")
            if total + len(text) > _settings.MAX_CONTEXT_CHARS:
                remaining = _settings.MAX_CONTEXT_CHARS - total
                if remaining > 0:
                    pieces.append(text[:remaining])
                break
            pieces.append(text)
            total += len(text)
        return "\n\n".join(pieces)

    def _build_prompt(self, history: List[Dict[str, str]], user_message: str, context: str) -> str:
        sys = (
            "You are a helpful AI assistant. Use the provided context to answer the user's query.\n"
            "If the answer is not in the context, say you don't know and suggest uploading relevant documents.\n"
        )
        hist_lines = []
        for turn in history[-10:]:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            hist_lines.append(f"{role.capitalize()}: {content}")

        return (
            f"{sys}\n"
            f"Context:\n{context}\n\n"
            f"Conversation so far:\n" + "\n".join(hist_lines) + "\n\n"
            f"User: {user_message}\n"
            f"Assistant:"
        )