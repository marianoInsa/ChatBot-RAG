from langchain_google_genai import ChatGoogleGenerativeAI

import logging
from app.config.config import get_settings
settings = get_settings()

logger = logging.getLogger(__name__)

def get_gemini(user_api_key: str | None = None) -> ChatGoogleGenerativeAI | None:
    api_key = user_api_key or settings.google_api_key
    if not api_key:
        logger.warning("No se proporcionó API Key para Google Gemini.")
        return None

    return ChatGoogleGenerativeAI(
        model=settings.google_model,
        api_key=api_key
    )

if __name__ == "__main__":
    print(get_gemini())