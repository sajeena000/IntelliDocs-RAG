from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.models.database import get_db
from app.models.schemas import ChatRequest, ChatResponse
from app.services.chat_memory_service import ChatMemoryService
from app.services.agent_service import UnifiedAgentService # <-- NEW

router = APIRouter(tags=["Chat"])

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    agent = UnifiedAgentService.get()
    memory = ChatMemoryService.get()

    history = memory.get_history(req.session_id)

    # The agent handles everything: RAG, tool use, and response generation
    agent_response = agent.answer(
        user_message=req.message,
        history=history,
        db=db,
    )

    # Save conversation to memory
    memory.append(req.session_id, role="user", content=req.message)
    memory.append(req.session_id, role="assistant", content=agent_response.reply)

    return agent_response