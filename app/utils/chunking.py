from __future__ import annotations
from typing import List
import nltk
from sentence_transformers import SentenceTransformer
import numpy as np

from app.core.config import get_settings
_settings = get_settings()


nltk.download("punkt")
nltk.download('punkt_tab')


def fixed_size_chunk(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Simple fixed-size chunking by characters with overlap.
    """
    chunks: List[str] = []
    i = 0
    n = len(text)
    if chunk_size <= 0:
        return [text]
    overlap = max(0, min(overlap, chunk_size - 1))
    while i < n:
        end = min(i + chunk_size, n)
        chunks.append(text[i:end])
        if end >= n:
            break
        i = end - overlap
    return chunks


def semantic_chunk(text: str, target_size: int = 500, overlap: int = 100, model: SentenceTransformer | None = None) -> List[str]:
    """
    Semantic chunking: sentence-based grouping with soft boundary by size and similarity proxy.
    We use sentence tokenization and then group sentences until the target size is reached.
    We also add overlap by repeating last portion from previous chunk.
    """
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return fixed_size_chunk(text, chunk_size=target_size, overlap=overlap)

    if model is None:
        model = SentenceTransformer(_settings.SEMANTIC_CHUNK_MODEL_PATH)

    sent_embs = model.encode(sentences, normalize_embeddings=True)

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    last_emb = None

    def add_chunk(lines: List[str]):
        if lines:
            chunks.append(" ".join(lines))

    for i, sent in enumerate(sentences):
        s = sent.strip()
        if not s:
            continue
        s_len = len(s)
        emb = sent_embs[i]

        if current_len == 0:
            current.append(s)
            current_len += s_len
            last_emb = emb
            continue

        sim = float(np.dot(last_emb, emb))

        should_split = (current_len >= target_size) or (current_len >= int(target_size * 0.7) and sim < 0.2)

        if should_split:
            add_chunk(current)

            if overlap > 0 and chunks:
                tail = chunks[-1][-overlap:]
                current = [tail, s]
                current_len = len(tail) + s_len
            else:
                current = [s]
                current_len = s_len
        else:
            current.append(s)
            current_len += s_len

        last_emb = emb

    add_chunk(current)
    return chunks