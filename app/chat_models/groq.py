from langchain_groq import ChatGroq

import logging
from app.config.config import get_settings

logger = logging.getLogger(__name__)

def get_groq() -> ChatGroq | None:
    if not get_settings().groq_api_key:
        return None

    return ChatGroq(
        model="qwen/qwen3-32b",
        temperature=0.1,
        max_tokens=1024
    )