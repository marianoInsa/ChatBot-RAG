from pydantic import BaseModel
from typing import Optional

class ChatQuestion(BaseModel):
    question: str
    # api_key: Optional[str] = None
  
class ChatResponse(BaseModel):
    response: str
    # extension a futuro
    # source_documents: Optional[list] = None
    # confidence_score: Optional[float] = None