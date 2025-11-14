from __future__ import annotations
from typing import Optional, List, Dict
from threading import Lock
from dataclasses import dataclass
import re

from sqlalchemy.orm import Session
from datetime import date as _date, time as _time

from google import genai
from google.genai import types

from app.core.config import get_settings
from app.models.models import Booking
from app.models.schemas import BookingCreate

_settings = get_settings()
_singleton_lock = Lock()


@dataclass
class BookingAttemptResult:
    needs_clarification: bool
    reply: str
    booking: Optional[BookingCreate] = None


def _is_iso_date(date_str: str) -> bool:
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", (date_str or "").strip()))


def _is_24h_time(time_str: str) -> bool:
    s = (time_str or "").strip()
    if not re.fullmatch(r"\d{2}:\d{2}", s):
        return False
    hh, mm = s.split(":")
    try:
        h = int(hh)
        m = int(mm)
        return 0 <= h <= 23 and 0 <= m <= 59
    except Exception:
        return False


def _looks_ambiguous_date(date_str: str) -> bool:
    if not date_str:
        return True
    lower = date_str.strip().lower()
    # relative or vague phrasing
    relative_terms = [
        "today", "tomorrow", "yesterday", "tonight",
        "next ", "this ", "coming ",
        "in ", "later", "soon", "end of", "start of",
    ]
    if any(t in lower for t in relative_terms):
        return True
    return not _is_iso_date(date_str)


def _looks_ambiguous_time(time_str: str) -> bool:
    if not time_str:
        return True
    lower = time_str.strip().lower()
    vague = ["morning", "afternoon", "evening", "noon", "midnight", "around"]
    if any(v in lower for v in vague):
        return True
    if re.search(r"\b(am|pm)\b", lower):
        return True
    return not _is_24h_time(time_str)


def _clarification_question(name: str, email: str, date_str: str, time_str: str) -> str:
    missing = []
    asks = []
    if not name:
        missing.append("full name")
    if not email:
        missing.append("email")
    if _looks_ambiguous_date(date_str):
        asks.append("an exact date in YYYY-MM-DD (e.g., 2025-03-15), not 'tomorrow' or 'next Friday'")
    if _looks_ambiguous_time(time_str):
        asks.append("an exact time in 24-hour HH:MM (e.g., 14:30), not '3pm' or 'evening'")

    parts = []
    if missing:
        parts.append(f"Please provide your {', '.join(missing)}.")
    if asks:
        parts.append("Also, please confirm " + " and ".join(asks) + ".")
    if not parts:
        return "Could you confirm the exact date (YYYY-MM-DD) and time (HH:MM, 24-hour)?"
    return " ".join(parts)


class BookingService:
    _instance: Optional["BookingService"] = None

    def __init__(self):
        self._model = "gemini-2.5-flash"

    @classmethod
    def get(cls) -> "BookingService":
        if cls._instance is None:
            with _singleton_lock:
                if cls._instance is None:
                    cls._instance = BookingService()
        return cls._instance

    def _client(self) -> Optional[genai.Client]:
        if not _settings.GOOGLE_API_KEY:
            return None
        return genai.Client(api_key=_settings.GOOGLE_API_KEY)
    
    @staticmethod
    def is_booking_intent(message: str) -> bool:
        if not message:
            return False
        m = message.lower()

        negative_phrases = [
            "what is", "what's", "how do", "how to", "how does",
            "explain", "tell me about", "docs", "documentation",
            "guide", "policy", "pricing", "price", "cost",
            "example", "sample", "tutorial", "booking.com",
        ]
        if any(p in m for p in negative_phrases):
            return False

        action_pattern = re.compile(
            r"\b(book|schedule|reserve|arrange|set\s*up|setup|make|create|confirm|reschedule|cancel|add|put)\b"
        )
        noun_pattern = re.compile(
            r"\b(appointment|interview|booking|meeting|slot|booking table)\b"
        )

        if "booking table" in m:
            return True

        if action_pattern.search(m) and noun_pattern.search(m):
            return True

        if re.search(r"\b(book|schedule|reserve)\s+me\b", m):
            return True

        return False

    def _create_booking_declaration(self) -> dict:
        
        return {
            "name": "create_booking",
            "description": (
                "Create a booking when all fields are explicit and unambiguous. "
                "Only call this when you have: name, email, date as YYYY-MM-DD, and time as HH:MM (24-hour). "
                "If any info is missing or ambiguous (e.g., 'tomorrow', 'next Friday', '3pm', 'evening'), "
                "ask a short clarifying question instead of calling the function."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Full name of the person."},
                    "email": {"type": "string", "description": "Email address of the person."},
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD."},
                    "time": {"type": "string", "description": "Time in HH:MM, 24-hour."},
                },
                "required": ["name", "email", "date", "time"],
            },
        }

    def _system_instruction(self) -> str:
        return (
            "You are an assistant that handles booking interview appointments using a function call:\n"
            "- Only call create_booking when name, email, date (YYYY-MM-DD), and time (HH:MM 24-hour) are explicit and unambiguous.\n"
            "- If dates/times are relative or vague (e.g., 'tomorrow', 'next Friday', '3pm', 'evening'), ask a concise clarifying question.\n"
            "- If no booking intent, respond normally.\n"
        )

    def _to_contents(self, history: List[Dict[str, str]], user_message: str) -> List[types.Content]:
        contents: List[types.Content] = []
        for turn in history[-10:]:
            role = turn.get("role", "user")
            text = turn.get("content", "") or ""
            if not text:
                continue
            contents.append(types.Content(role="user" if role == "user" else "model", parts=[types.Part(text=text)]))
        contents.append(types.Content(role="user", parts=[types.Part(text=user_message)]))
        return contents

    def try_create_booking(
        self,
        session_id: str,
        message: str,
        history: List[Dict[str, str]],
        model: str = "gemini",
    ) -> Optional[BookingAttemptResult]:
        
        if model.strip().lower() != "gemini":
            return None
        if not self.is_booking_intent(message):
            print("[BookingService] Message does not express booking intent.")
            return None

        client = self._client()
        if client is None:
            return None

        # Define tools per docs
        create_booking_fn = self._create_booking_declaration()
        tools = types.Tool(function_declarations=[create_booking_fn])
        config = types.GenerateContentConfig(
            tools=[tools],
            temperature=0.2,
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="AUTO")
            ),
            system_instruction=self._system_instruction(),
        )

        contents = self._to_contents(history, message)

        try:
            response = client.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            print(f"[BookingService] generate_content error: {e}")
            return None

        candidate = (response.candidates or [None])[0]
        if not candidate or not getattr(candidate, "content", None):
            return None

        parts = getattr(candidate.content, "parts", []) or []
        function_call = None
        for p in parts:
            if getattr(p, "function_call", None):
                function_call = p.function_call
                break

        if not function_call:
            text = (response.text or "").strip()
            if text:
                return BookingAttemptResult(needs_clarification=True, reply=text, booking=None)
            return None

        if function_call.name != "create_booking":
            return None

        args = dict(function_call.args or {})
        name = (args.get("name") or "").strip()
        email = (args.get("email") or "").strip()
        date_str = (args.get("date") or "").strip()
        time_str = (args.get("time") or "").strip()

        # Validate unambiguously
        needs_ask = False
        if not name or not email:
            needs_ask = True
        if _looks_ambiguous_date(date_str):
            needs_ask = True
        if _looks_ambiguous_time(time_str):
            needs_ask = True

        if needs_ask:
            question = _clarification_question(name, email, date_str, time_str)
            return BookingAttemptResult(needs_clarification=True, reply=question, booking=None)
        try:
            booking = BookingCreate(name=name, email=email, date=date_str, time=time_str)
        except Exception:
            question = (
                "I couldn't parse the booking details. Please provide full name, email, date in YYYY-MM-DD, "
                "and time in HH:MM (24-hour)."
            )
            return BookingAttemptResult(needs_clarification=True, reply=question, booking=None)

        return BookingAttemptResult(needs_clarification=False, reply="", booking=booking)

    def save_booking(self, db: Session, booking: BookingCreate) -> Booking:
        bdate = booking.date
        if isinstance(bdate, str):
            bdate = _date.fromisoformat(bdate)
        hh, mm = booking.time.split(":")
        tm = _time(hour=int(hh), minute=int(mm))

        row = Booking(name=booking.name, email=booking.email, date=bdate, time=tm)
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
        """
        Step 4 from docs: Create a user-friendly response by sending the function result back to the model.
        """
        client = self._client()
        if client is None:
            return ""

        function_response_part = types.Part.from_function_response(
            name=function_name,
            response={"result": function_result},
        )

        contents = list(original_contents)
        contents.append(model_raw_response_content)  # append the model's previous response content
        contents.append(types.Content(role="user", parts=[function_response_part]))  # append function response

        try:
            final_response = client.models.generate_content(
                model=self._model,
                contents=contents,
                config=types.GenerateContentConfig(temperature=0.2),
            )
            return (final_response.text or "").strip()
        except Exception as e:
            print(f"[BookingService] finalize reply error: {e}")
            return ""
