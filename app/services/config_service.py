from typing import Optional
from langchain_core.embeddings import Embeddings

from app.config.config import get_settings
from app.embedding_models.factory import get_embeddings
from app.models.client_models import ClientConfig

import logging

logger = logging.getLogger(__name__)


def get_client_config(client_config: Optional[ClientConfig] = None) -> dict:
    """
    Mergea la configuración del cliente con los valores por defecto globales.
    Los valores del cliente tienen prioridad sobre los defaults.
    """
    settings = get_settings()
    defaults = {
        "embedding_provider": "huggingface",
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "mmr_k": settings.mmr_k,
        "mmr_fetch_k": settings.mmr_fetch_k,
        "mmr_lambda_mult": settings.mmr_lambda_mult,
        "max_context_length": settings.max_context_length,
    }
    if client_config is None:
        return defaults
    config_dict = client_config.model_dump(exclude_none=True)
    return {**defaults, **config_dict}


def get_embeddings_for_client(config: dict) -> Optional[Embeddings]:
    """
    Obtiene el modelo de embeddings según la configuración del cliente.
    """
    provider = config.get("embedding_provider", "huggingface")
    if provider == "gemini":
        return get_embeddings("gemini")
    return get_embeddings("default")
