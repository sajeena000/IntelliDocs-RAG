from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional
from threading import Lock
from datetime import date, datetime

from sqlalchemy.orm import Session
from google.genai import types, client

from app.core.config import get_settings
from app.models.schemas import ChatResponse, SourceChunk, BookingCreate
from app.services.vector_service import VectorService
from app.services.bm25_service import BM25Service
from app.services.booking_service import BookingService, _looks_ambiguous_date, _looks_ambiguous_time, _clarification_question
from app.utils.reranker import Reranker

_settings = get_settings()
_singleton_lock = Lock()

class UnifiedAgentService:
    _instance: Optional["UnifiedAgentService"] = None

    def __init__(self):
        self.vector = VectorService.get()
        self.bm25 = BM25Service.get()
        self.reranker = Reranker.get()
        self.booking_service = BookingService.get()
        self._model_name = "gemini-2.5-flash"
        
    @classmethod
    def get(cls) -> "UnifiedAgentService":
        if cls._instance is None:
            with _singleton_lock:
                if cls._instance is None:
                    cls._instance = UnifiedAgentService()
        return cls._instance

    def _build_system_prompt(self, context: str) -> str:
        today = date.today().isoformat()
        return (
            "You are an intelligent assistant. You have access to two resources:\n"
            "1. A Knowledge Base (provided as 'Context').\n"
            "2. A Booking Tool (function: create_booking).\n\n"
            "YOUR GOAL: Help the user by answering questions using the Context or creating bookings using the Tool.\n\n"
            "RULES FOR BEHAVIOR:\n"
            "1. **Analyze Intent**: If the user asks for info (hours, policies), use the Context. If they want to schedule, use the Booking Tool. If they do both, answer the question from the context first, then proceed with the booking in the same response if possible.\n"
            "2. **Use Context**: Only answer based on the provided Context. If the answer isn't there, say you don't know. Check the Context for booking constraints (e.g., 'Closed on Sundays') before calling the tool.\n"
            "3. **Use Booking Tool**: You must have all 4 parameters: name, email, date (YYYY-MM-DD), and time (HH:MM). If any are missing or ambiguous (e.g., 'tomorrow', day before yesterday, thursday), DO NOT call the tool. Instead, ask a clarifying question.\n"
            "4. **Be Proactive**: If a user asks a question that can be answered from context but also hints at booking, answer the question and then ask if they would like to book an appointment.\n\n"
            f"**Today's Date is: {today}**\n\n"
            "---BEGIN CONTEXT---\n"
            f"{context}\n"
            "---END CONTEXT---"
        )
        
    def _retrieve_and_rerank_context(self, user_message: str, db: Session) -> List[Dict]:
        # Dense search
        q_vec = self.vector.get_embeddings([user_message])[0]
        dense_hits = self.vector.search(q_vec, limit=_settings.TOP_K_DENSE)

        # BM25 search
        if self.bm25._bm25 is None: self.bm25.rebuild(db)
        bm25_hits = self.bm25.search(user_message, top_k=_settings.TOP_K_BM25)

        # Fetch BM25 payloads
        chunk_map = {}
        if bm25_hits:
            from app.models.models import Chunk
            chunk_ids = [h["chunk_id"] for h in bm25_hits]
            rows = db.query(Chunk).filter(Chunk.id.in_(chunk_ids)).all()
            for r in rows:
                chunk_map[str(r.id)] = { "document_id": str(r.document_id), "chunk_id": str(r.id), "chunk_index": r.chunk_index, "text": r.text }
        bm25_payloads = [ {**meta, "score": h["score"]} for h in bm25_hits if (meta := chunk_map.get(h["chunk_id"])) ]

        # Merge and rerank
        candidates = {item["chunk_id"]: item for item in dense_hits}
        for item in bm25_payloads:
            if item["chunk_id"] in candidates:
                candidates[item["chunk_id"]]["score"] = max(float(candidates[item["chunk_id"]]["score"]), float(item["score"]))
            else:
                candidates[item["chunk_id"]] = item
        
        reranked = self.reranker.rerank(user_message, list(candidates.values()))
        return reranked[:_settings.TOP_K_FINAL]

    def _format_context_for_prompt(self, chunks: List[Dict]) -> str:
        pieces = [ch.get("text", "") for ch in chunks]
        return "\n\n".join(pieces)[:_settings.MAX_CONTEXT_CHARS]

    def _format_chat_history(self, history: List[Dict[str, str]]) -> List[types.Content]:
        contents: List[types.Content] = []
        for turn in history[-10:]:
            role = "user" if turn.get("role") == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part(text=turn.get("content", ""))] ))
        return contents
    
    def answer(self, user_message: str, history: List[Dict[str, str]], db: Session) -> ChatResponse:
        top_chunks = self._retrieve_and_rerank_context(user_message, db)
        context_text = self._format_context_for_prompt(top_chunks)

        gemini_client = self.booking_service._client()
        if not gemini_client:
            return ChatResponse(reply="Error: Gemini client not configured.", sources=[], booking_created=False)

        system_prompt = self._build_system_prompt(context_text)
        tools = types.Tool(function_declarations=[self.booking_service._create_booking_declaration()])
        
        contents = self._format_chat_history(history)
        contents.append(types.Content(role="user", parts=[types.Part(text=user_message)]))
        
        config = types.GenerateContentConfig(
            tools=[tools],
            temperature=0.2,
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="AUTO")
            ),
            system_instruction=system_prompt,
        )

        try:
            response = gemini_client.models.generate_content(
                model=self._model_name,
                contents=contents,
                config=config,
            )
        except Exception as e:
            print(f"[AgentService] Gemini API error: {e}")
            return ChatResponse(reply="Sorry, I encountered an error. Please try again.", sources=[], booking_created=False)

        candidate = (response.candidates or [None])[0]
        if not candidate or not candidate.content or not candidate.content.parts:
            return ChatResponse(reply="I'm sorry, I could not generate a response.", sources=[], booking_created=False)
        
        part = candidate.content.parts[0]
        if part.function_call and part.function_call.name == "create_booking":
            args = dict(part.function_call.args or {})
            name, email, date_str, time_str = (args.get(k, "").strip() for k in ["name", "email", "date", "time"])

            if not all([name, email]) or _looks_ambiguous_date(date_str) or _looks_ambiguous_time(time_str):
                clarification = _clarification_question(name, email, date_str, time_str)
                return ChatResponse(reply=clarification, sources=[], booking_created=False)

            try:
                booking_data = BookingCreate(name=name, email=email, date=date_str, time=time_str)
                booking_row = self.booking_service.save_booking(db, booking_data)
                
                fn_result = {"booking_id": str(booking_row.id), "name": booking_row.name, "email": booking_row.email, "date": str(booking_row.date), "time": booking_row.time.strftime("%H:%M")}
                
                final_reply = self.booking_service.finalize_reply_with_function_result(
                    original_contents=contents,
                    model_raw_response_content=candidate.content,
                    function_name="create_booking",
                    function_result=fn_result
                )
                
                return ChatResponse(reply=final_reply, sources=[], booking_created=True, booking_id=booking_row.id)
            except Exception as e:
                print(f"[AgentService] Booking creation error: {e}")
                return ChatResponse(reply="I couldn't finalize the booking due to an internal error.", sources=[], booking_created=False)

        reply_text = response.text or "I am unable to provide a response at this time."
        sources = [SourceChunk(document_id=s["document_id"], chunk_index=s["chunk_index"], text_preview=(s["text"][:200] + "...")) for s in top_chunks]
        return ChatResponse(reply=reply_text, sources=sources, booking_created=False)