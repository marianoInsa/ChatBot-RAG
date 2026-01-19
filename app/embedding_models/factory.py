from langchain_core.embeddings import Embeddings
from app.embedding_models.huggingface import hugging_face_embeddings
from app.embedding_models.gemini import get_gemini_embeddings
import logging
logger = logging.getLogger(__name__)

def get_embeddings(embeddings: str) -> Embeddings | None:
    if embeddings == "gemini":
        return get_gemini_embeddings()
    elif embeddings == "huggingface":
        return hugging_face_embeddings
    else:
        logger.error("Modelo de embeddings no soportado o error al cargar el modelo.")
        return None