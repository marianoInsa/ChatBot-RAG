from langchain_google_genai import ChatGoogleGenerativeAI

import logging
from app.config.config import get_settings

logger = logging.getLogger(__name__)

def get_gemini() -> ChatGoogleGenerativeAI | None:
    if not get_settings().google_api_key:
        return None

    return ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

if __name__ == "__main__":
    print(get_gemini())