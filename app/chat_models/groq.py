from langchain_groq import ChatGroq

import logging
from app.config.config import get_settings
settings = get_settings()

logger = logging.getLogger(__name__)

def get_groq(user_api_key: str | None = None) -> ChatGroq | None:
    api_key = user_api_key or settings.groq_api_key
    if not api_key:
        logger.warning("No se proporcionó API Key para Groq.")
        return None

    return ChatGroq(
        api_key=api_key,
        model=settings.groq_model,
        temperature=0.1,
        max_tokens=1024
    )