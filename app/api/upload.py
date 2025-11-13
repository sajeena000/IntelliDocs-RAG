import io
import os
import uuid
from typing import List, Tuple

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.database import get_db
from app.models.models import Document, Chunk
from app.models.schemas import IngestResponse, IngestBatchResponse
from app.services.vector_service import VectorService
from app.services.document_service import extract_text_from_pdf, extract_text_from_txt
from app.utils.chunking import fixed_size_chunk, semantic_chunk
from app.services.bm25_service import BM25Service

router = APIRouter(tags=["Ingestion"])

settings = get_settings()


@router.post("/ingest", response_model=IngestBatchResponse)
def ingest_documents(
    files: List[UploadFile] = File(...),
    chunking_strategy: str = Form("fixed", description="fixed or semantic"),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(100),
    db: Session = Depends(get_db),
) -> IngestBatchResponse:
    """
    Upload .pdf or .txt, extract text, chunk, embed, store in Qdrant, and save metadata in Postgres.
    """
    if chunking_strategy not in {"fixed", "semantic"}:
        raise HTTPException(status_code=400, detail="chunking_strategy must be 'fixed' or 'semantic'")

    vector_service = VectorService.get()
    bm25_service = BM25Service.get()

    results: List[IngestResponse] = []

    for file in files:
        filename = file.filename or "uploaded"
        content_type = (file.content_type or "").lower()

        if filename.lower().endswith(".pdf") or "pdf" in content_type:
            data = file.file.read()
            text = extract_text_from_pdf(io.BytesIO(data))
            content_type = "application/pdf"
        elif filename.lower().endswith(".txt") or "text" in content_type:
            data = file.file.read()
            text = extract_text_from_txt(io.BytesIO(data))
            content_type = "text/plain"
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {filename}")

        if not text.strip():
            raise HTTPException(status_code=400, detail=f"No extractable text in {filename}")

        # Create Document
        doc = Document(title=os.path.splitext(filename)[0], filename=filename, content_type=content_type)
        db.add(doc)
        db.commit()
        db.refresh(doc)

        # Chunking
        if chunking_strategy == "fixed":
            chunks = fixed_size_chunk(text, chunk_size=chunk_size, overlap=chunk_overlap)
        else:
            chunks = semantic_chunk(text, target_size=chunk_size, overlap=chunk_overlap)

        # Persist chunks
        chunk_rows: List[Chunk] = []
        for idx, ch in enumerate(chunks):
            chunk_rows.append(Chunk(document_id=doc.id, chunk_index=idx, text=ch))
        db.add_all(chunk_rows)
        db.commit()
        for ch in chunk_rows:
            db.refresh(ch)

        # Embed + upsert to Qdrant
        payloads = []
        vectors = VectorService.get_embeddings([c.text for c in chunk_rows])
        point_ids: List[str] = []
        for i, ch in enumerate(chunk_rows):
            pid = str(uuid.uuid4())
            point_ids.append(pid)
            payloads.append(
                {
                    "document_id": str(ch.document_id),
                    "chunk_id": str(ch.id),
                    "chunk_index": ch.chunk_index,
                    "text": ch.text,
                }
            )

        vector_service.upsert_points(point_ids, vectors, payloads)

        # Update qdrant ids in DB
        for ch, pid in zip(chunk_rows, point_ids):
            ch.qdrant_point_id = pid
        db.add_all(chunk_rows)
        db.commit()

        results.append(IngestResponse(document_id=doc.id, chunks_indexed=len(chunk_rows)))

    # Rebuild BM25 index after ingestion
    bm25_service.rebuild(db)

    return IngestBatchResponse(results=results)