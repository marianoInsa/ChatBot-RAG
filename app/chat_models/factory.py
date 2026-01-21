from langchain_core.language_models.chat_models import BaseChatModel
from app.chat_models.groq import get_groq
from app.chat_models.gemini import get_gemini
from app.chat_models.ollama import get_ollama_instance
from app.config.config import get_settings
settings = get_settings()
import logging
logger = logging.getLogger(__name__)

def get_chat_model(chat_model: str = "ollama", user_api_key: str | None = None) -> BaseChatModel | None:
    if chat_model == "ollama":
        if settings.enable_ollama:
            return get_ollama_instance()
        else:
            logger.warning("Ollama sólo está habilitado localmente.")
            return None

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