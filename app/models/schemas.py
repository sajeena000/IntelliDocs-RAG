from __future__ import annotations
import uuid
from datetime import date, time
from typing import List, Optional
from pydantic import BaseModel, Field, EmailStr


class IngestResponse(BaseModel):
    document_id: uuid.UUID
    chunks_indexed: int


class IngestBatchResponse(BaseModel):
    results: List[IngestResponse]


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique chat session ID (e.g., user ID or UUID).")
    message: str
    model: str = Field(default="gemini", description="Model choice: 'gemini' or 'local'.")


class SourceChunk(BaseModel):
    document_id: uuid.UUID
    chunk_index: int
    text_preview: str


class ChatResponse(BaseModel):
    reply: str
    sources: List[SourceChunk] = Field(default_factory=list)
    booking_created: bool = False
    booking_id: Optional[uuid.UUID] = None


class BookingCreate(BaseModel):
    name: str
    email: EmailStr
    date: date
    time: str  # "HH:MM" 24-hour format


class BookingResponse(BaseModel):
    booking_id: uuid.UUID
    name: str
    email: EmailStr
    date: date
    time: str  # "HH:MM" 24-hour format
    
class BookingInfo(BaseModel):
    name: Optional[str] = Field(None, description="The full name of the person making the booking.")
    email: Optional[EmailStr] = Field(None, description="The email address for the booking confirmation.")
    date: Optional[str] = Field(None, description="The date of the appointment in YYYY-MM-DD format.")
    time: Optional[str] = Field(None, description="The time of the appointment in HH:MM (24-hour) format.")