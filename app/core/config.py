import os
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # App
    ENV: str = Field(default="development")
    LOG_LEVEL: str = Field(default="INFO")

    # Database (Postgres)
    POSTGRES_HOST: str = Field(default="postgres")
    POSTGRES_PORT: int = Field(default=5432)
    POSTGRES_DB: str = Field(default="postgres")
    POSTGRES_USER: str = Field(default="postgres")
    POSTGRES_PASSWORD: str = Field(default="postgres")

    # Redis
    REDIS_HOST: str = Field(default="redis")
    REDIS_PORT: int = Field(default=6379)
    REDIS_DB: int = Field(default=0)
    REDIS_CHAT_HISTORY_TTL_SECONDS: int = Field(default=60 * 60 * 24 * 7)  # 7 days
    REDIS_MAX_TURNS: int = Field(default=20)

    # Qdrant
    QDRANT_HOST: str = Field(default="qdrant")
    QDRANT_PORT: int = Field(default=6333)
    QDRANT_COLLECTION: str = Field(default="documents")

    # Embeddings
    EMBEDDING_MODEL_PATH: str = Field(default="/app/models/embeddinggemma-300m")
    RERANKER_MODEL_PATH: str = Field(default="/app/models/ms-marco-MiniLM-L6-v2")
    SEMANTIC_CHUNK_MODEL_PATH: str = Field(default="/app/models/all-MiniLM-L6-v2")
    
    # LLMs
    # "gemini" or "local"
    DEFAULT_LLM: str = Field(default="gemini")

    # Gemini
    GOOGLE_API_KEY: str = Field(default=os.getenv("GOOGLE_API_KEY", ""))

    # Local LLM (llama.cpp)
    LLAMA_MODEL_PATH: str = Field(default=os.getenv("LLAMA_MODEL_PATH", "")) 
    LLAMA_CTX_SIZE: int = Field(default=4096)
    LLAMA_N_THREADS: int = Field(default=4)
    LLAMA_N_GPU_LAYERS: int = Field(default=0)  

    # RAG settings
    TOP_K_DENSE: int = Field(default=20)
    TOP_K_BM25: int = Field(default=20)
    TOP_K_FINAL: int = Field(default=5)
    MAX_CONTEXT_CHARS: int = Field(default=6000)

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()