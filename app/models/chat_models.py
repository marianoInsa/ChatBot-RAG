from pydantic import BaseModel
from typing import Optional, Literal

class ChatQuestion(BaseModel):
    question: str
    api_key: str = ""
    model_provider: Literal["ollama", "groq", "gemini"] = "ollama"
  
class ChatResponse(BaseModel):
    response: str
    # extension a futuro
    # source_documents: Optional[list] = None
    # confidence_score: Optional[float] = None