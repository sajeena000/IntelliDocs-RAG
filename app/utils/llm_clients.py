from __future__ import annotations
from typing import Dict, Optional
from threading import Lock
import json
from app.models.schemas import BookingInfo
from pydantic_core import ValidationError
import google.genai as genai

from app.core.config import get_settings

_settings = get_settings()
_singleton_lock = Lock()

class BaseLLMClient:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError

class GeminiClient(BaseLLMClient):
    def __init__(self):
        
        if not _settings.GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY is not set for Gemini.")
        
        self.client = genai.Client(api_key=_settings.GOOGLE_API_KEY)
        self._model = "gemini-2.5-flash"
        
    def generate(self, prompt: str) -> str:
        try:
            resp = self.client.models.generate_content(model=self._model, contents=prompt)
        except Exception as e:
            print(f"Unexpected error during API call: {e}")
            resp = None
        try:
            return resp.text
        except (ValueError, IndexError):
            return ""
        
class LlamaCppClient(BaseLLMClient):
    def __init__(self):
        from llama_cpp import Llama

        if not _settings.LLAMA_MODEL_PATH:
            raise RuntimeError("LLAMA_MODEL_PATH is not set for local LLM.")
        self._llm = Llama(
            model_path=_settings.LLAMA_MODEL_PATH,
            n_ctx=_settings.LLAMA_CTX_SIZE,
            n_threads=_settings.LLAMA_N_THREADS,
            n_gpu_layers=_settings.LLAMA_N_GPU_LAYERS,
            verbose=False,
        )

    def generate(self, prompt: str) -> str:
        out = self._llm(
            prompt,
            max_tokens=1024,
            temperature=0.3,
            top_p=0.9,
            repeat_penalty=1.1,
            stop=["</s>", "User:", "Assistant:"],
        )
        return out.get("choices", [{}])[0].get("text", "").strip()


class LLMRouter:
    _gemini: Optional[GeminiClient] = None
    _local: Optional[LlamaCppClient] = None

    @classmethod
    def get(cls, model: str = "gemini") -> BaseLLMClient:
        m = model.strip().lower()
        if m == "gemini":
            if cls._gemini is None:
                with _singleton_lock:
                    if cls._gemini is None:
                        cls._gemini = GeminiClient()
            return cls._gemini
        elif m == "local":
            if cls._local is None:
                with _singleton_lock:
                    if cls._local is None:
                        cls._local = LlamaCppClient()
            return cls._local
        else:
            # default fallback
            return cls.get(_settings.DEFAULT_LLM)