from langchain_core.language_models.chat_models import BaseChatModel
from app.chat_models.groq import get_groq
from app.chat_models.gemini import get_gemini
from app.chat_models.ollama import ollama
import logging
logger = logging.getLogger(__name__)

def get_chat_model(chat_model: str = "ollama", user_api_key: str | None = None) -> BaseChatModel | None:
    if chat_model == "ollama":
        return ollama

    elif chat_model == "gemini":
        gemini = get_gemini(user_api_key)
        if gemini is not None:
            return gemini
    
    elif chat_model == "groq":
        groq = get_groq(user_api_key)
        if groq is not None:
            return groq

    logger.warning(f"No se pudo inicializar el modelo: {chat_model}")
    return None