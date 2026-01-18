from langchain_google_genai import GoogleGenerativeAIEmbeddings

import logging
from app.config.config import get_settings

logger = logging.getLogger(__name__)

def get_gemini_embeddings() -> GoogleGenerativeAIEmbeddings | None:
    if not get_settings().google_api_key:
        return None

    return GoogleGenerativeAIEmbeddings(model=get_settings().gemini_embeddings_model_name)