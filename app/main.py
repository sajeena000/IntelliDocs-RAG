from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import Settings, get_settings
from app.models.database import Base, engine
from app.api.upload import router as upload_router
from app.api.chat import router as chat_router

app = FastAPI(
    title="RAG API",
    description="API for document ingestion and conversational RAG.",
    version="1.0.0",
)

# CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup") 
def on_startup() -> None:
    
    Base.metadata.create_all(bind=engine)

    
    _ = get_settings() # Initialize settings on startup

app.include_router(upload_router, prefix="/api")
app.include_router(chat_router, prefix="/api")