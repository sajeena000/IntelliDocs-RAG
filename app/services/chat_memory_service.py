from __future__ import annotations
import json
from typing import List, Dict, Optional
from threading import Lock
import redis

from app.core.config import get_settings

_settings = get_settings()
_singleton_lock = Lock()


class ChatMemoryService:
    _instance: Optional["ChatMemoryService"] = None

    def __init__(self):
        self._client = redis.Redis(
            host=_settings.REDIS_HOST,
            port=_settings.REDIS_PORT,
            db=_settings.REDIS_DB,
            decode_responses=True,
        )

    @classmethod
    def get(cls) -> "ChatMemoryService":
        if cls._instance is None:
            with _singleton_lock:
                if cls._instance is None:
                    cls._instance = ChatMemoryService()
        return cls._instance

    def _key(self, session_id: str) -> str:
        return f"chat:{session_id}"

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        key = self._key(session_id)
        data = self._client.get(key)
        if not data:
            return []
        try:
            history = json.loads(data)
            return history
        except Exception:
            return []

    def append(self, session_id: str, role: str, content: str) -> None:
        key = self._key(session_id)
        history = self.get_history(session_id)
        history.append({"role": role, "content": content})
        # trim
        if len(history) > _settings.REDIS_MAX_TURNS * 2:
            history = history[-_settings.REDIS_MAX_TURNS * 2 :]
        self._client.setex(key, _settings.REDIS_CHAT_HISTORY_TTL_SECONDS, json.dumps(history))

    def clear(self, session_id: str) -> None:
        self._client.delete(self._key(session_id))