from langchain_core.language_models.chat_models import BaseChatModel
from app.chat_models.groq import get_groq
from app.chat_models.gemini import get_gemini
# from app.chat_models.ollama import llama2
import logging
logger = logging.getLogger(__name__)

def get_chat_model(chat_model: str | None) -> BaseChatModel | None:
    if chat_model == "gemini":
        gemini = get_gemini()
        if gemini is not None:
            return gemini
    
    elif chat_model == "groq":
        groq = get_groq()
        if groq is not None:
            return groq
    
    # elif chat_model == "llama2":
        # return llama2

    else:
        logger.warning("Modelo de chat no soportado o error al cargar el modelo.")
        return None