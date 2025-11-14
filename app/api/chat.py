from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.database import get_db
from app.models.schemas import ChatRequest, ChatResponse, SourceChunk
from app.services.rag_service import RAGService
from app.services.chat_memory_service import ChatMemoryService
from app.services.booking_service import BookingService
from google.genai import types

router = APIRouter(tags=["Chat"])

settings = get_settings()


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    rag = RAGService.get()
    memory = ChatMemoryService.get()
    booking_service = BookingService.get()

    history = memory.get_history(req.session_id)

    # Gated booking attempt: only if message clearly expresses booking intent
    if booking_service.is_booking_intent(req.message):
        print("[chat] Booking intent detected, attempting booking flow.")
        booking_attempt = booking_service.try_create_booking(
            session_id=req.session_id,
            message=req.message,
            history=history,
            model=req.model,
        )

        if booking_attempt is not None:
            if booking_attempt.needs_clarification:
                memory.append(req.session_id, role="user", content=req.message)
                memory.append(req.session_id, role="assistant", content=booking_attempt.reply)
                return ChatResponse(
                    reply=booking_attempt.reply,
                    sources=[],
                    booking_created=False,
                    booking_id=None,
                )

            if booking_attempt.booking is not None:
                booking_row = booking_service.save_booking(db, booking_attempt.booking)

                contents = []
                for turn in history[-10:]:
                    role = turn.get("role", "user")
                    text = turn.get("content", "") or ""
                    if text:
                        contents.append(types.Content(role="user" if role == "user" else "model", parts=[types.Part(text=text)]))
                contents.append(types.Content(role="user", parts=[types.Part(text=req.message)]))

                model_content = types.Content(role="model", parts=[types.Part(text="")])
                fn_result = {
                    "booking_id": str(booking_row.id),
                    "name": booking_row.name,
                    "email": booking_row.email,
                    "date": str(booking_row.date),
                    "time": booking_row.time.strftime("%H:%M"),
                }
                friendly = booking_service.finalize_reply_with_function_result(
                    original_contents=contents,
                    model_raw_response_content=model_content,
                    function_name="create_booking",
                    function_result=fn_result,
                )

                reply = friendly or (
                    f"Booking confirmed for {booking_row.name} on {booking_row.date} at {booking_row.time}. "
                    "A confirmation has been recorded."
                )

                memory.append(req.session_id, role="user", content=req.message)
                memory.append(req.session_id, role="assistant", content=reply)
                return ChatResponse(
                    reply=reply,
                    sources=[],
                    booking_created=True,
                    booking_id=booking_row.id,
                )
        # If booking intent but no function call and no clarification -> fall through to RAG

    # RAG pipeline
    result = rag.answer(
        user_message=req.message,
        session_id=req.session_id,
        history=history,
        model=req.model,
        db=db,
    )

    memory.append(req.session_id, role="user", content=req.message)
    memory.append(req.session_id, role="assistant", content=result.reply)

    sources: List[SourceChunk] = []
    for s in result.sources:
        sources.append(
            SourceChunk(
                document_id=s["document_id"],
                chunk_index=s["chunk_index"],
                text_preview=(s["text"][:200] + "..." if len(s["text"]) > 200 else s["text"]),
            )
        )

    return ChatResponse(reply=result.reply, sources=sources, booking_created=False, booking_id=None)