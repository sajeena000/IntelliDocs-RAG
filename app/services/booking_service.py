from __future__ import annotations
from typing import Optional, List, Dict
from threading import Lock
from dataclasses import dataclass
import re
from datetime import date as _date, time as _time, datetime

from sqlalchemy.orm import Session
from datetime import date as _date, time as _time

from google import genai
from google.genai import types

from app.core.config import get_settings
from app.models.models import Booking
from app.models.schemas import BookingCreate

_settings = get_settings()
_singleton_lock = Lock()

def _is_iso_date(date_str: str) -> bool:
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", (date_str or "").strip()))

def _is_24h_time(time_str: str) -> bool:
    s = (time_str or "").strip()
    if not re.fullmatch(r"\d{2}:\d{2}", s):
        return False
    hh, mm = s.split(":")
    try:
        h, m = int(hh), int(mm)
        return 0 <= h <= 23 and 0 <= m <= 59
    except Exception: return False

def _looks_ambiguous_date(date_str: str) -> bool:
    if not date_str: return True
    return not _is_iso_date(date_str)

def _looks_ambiguous_time(time_str: str) -> bool:
    if not time_str: return True
    lower = time_str.strip().lower()
    if any(v in lower for v in ["am", "pm", "morning", "afternoon", "evening"]): return True
    return not _is_24h_time(time_str)

def _clarification_question(name: str, email: str, date_str: str, time_str: str) -> str:
    missing, asks = [], []
    if not name: missing.append("full name")
    if not email: missing.append("email")
    if _looks_ambiguous_date(date_str): asks.append("an exact date in YYYY-MM-DD format (e.g., 2025-03-15)")
    if _looks_ambiguous_time(time_str): asks.append("an exact time in 24-hour HH:MM format (e.g., 14:30)")
    parts = []
    if missing: parts.append(f"Please provide your {', '.join(missing)}.")
    if asks: parts.append("Also, please confirm " + " and ".join(asks) + ".")
    return " ".join(parts) if parts else "Could you confirm the exact date (YYYY-MM-DD) and time (HH:MM, 24-hour)?"


class BookingService:
    _instance: Optional["BookingService"] = None

    def __init__(self):
        self._model_name = "gemini-2.5-flash"

    @classmethod
    def get(cls) -> "BookingService":
        if cls._instance is None:
            with _singleton_lock:
                if cls._instance is None:
                    cls._instance = BookingService()
        return cls._instance

    def _client(self) -> Optional[genai.client.Client]:
        if not _settings.GOOGLE_API_KEY:
            return None
        return genai.Client(api_key=_settings.GOOGLE_API_KEY)

    def _create_booking_declaration(self) -> dict:
        return {
            "name": "create_booking",
            "description": (
                "Creates a booking appointment in the system. "
                "Only call this when you have all four required parameters: name, email, date as YYYY-MM-DD, and time as HH:MM (24-hour). "
                "If any information is missing or ambiguous (e.g., 'tomorrow', 'next week', '3pm'), "
                "ask a short clarifying question instead of calling this function."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Full name of the person."},
                    "email": {"type": "string", "description": "Email address of the person."},
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format."},
                    "time": {"type": "string", "description": "Time in HH:MM, 24-hour format."},
                },
                "required": ["name", "email", "date", "time"],
            },
        }

    def save_booking(self, db: Session, booking: BookingCreate) -> Booking:
        bdate = _date.fromisoformat(booking.date) if isinstance(booking.date, str) else booking.date
        hh, mm = booking.time.split(":")
        tm = _time(hour=int(hh), minute=int(mm))

        row = Booking(name=booking.name, email=booking.email, date=bdate, time=tm, created_at=datetime.utcnow())
        db.add(row)
        db.commit()
        db.refresh(row)
        return row

    def finalize_reply_with_function_result(
        self,
        original_contents: List[types.Content],
        model_raw_response_content: types.Content,
        function_name: str,
        function_result: dict,
    ) -> str:
        client = self._client()
        if client is None: return ""

        function_response_part = types.Part.from_function_response(
            name=function_name,
            response={"result": function_result},
        )
        
        contents = list(original_contents)
        contents.append(model_raw_response_content)
        contents.append(types.Content(role="user", parts=[function_response_part]))

        try:
            final_response = client.models.generate_content(
                model=self._model_name,
                contents=contents,
            )
            return (final_response.text or "").strip()
        except Exception as e:
            print(f"[BookingService] finalize reply error: {e}")
            return "Booking confirmed."