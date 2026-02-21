import threading
import uuid
from datetime import datetime
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from app.models.client_models import ClientConfig
from app.services.config_service import get_client_config, get_embeddings_for_client
from app.services.data_service import DataIngestionService
from app.services.vector_store_cache import VectorStoreCache

import logging

logger = logging.getLogger(__name__)


class ClientManager:
    """
    Gestiona clientes multi-tenant: metadatos en memoria, vector stores en caché LRU (RAM).
    No hay persistencia en disco ni en la nube: los índices FAISS viven exclusivamente
    en el caché LRU. Al reiniciar el servidor o al ser desalojado por política LRU,
    el usuario debe recargar sus documentos.
    """

    def __init__(self, cache: VectorStoreCache):
        self._cache = cache
        self._metadata: dict[str, dict] = {}
        self._lock = threading.Lock()

    def register_client(self, config: Optional[ClientConfig] = None) -> str:
        """Registra un nuevo cliente. Retorna client_id."""
        with self._lock:
            client_id = str(uuid.uuid4())
            merged_config = get_client_config(config)
            self._metadata[client_id] = {
                "client_id": client_id,
                "created_at": datetime.utcnow(),
                "config": merged_config,
                "stats": {
                    "documents_count": 0,
                    "chunks_count": 0,
                    "last_updated": None,
                },
            }
            return client_id

    def get_client(self, client_id: str) -> Optional[dict]:
        """Obtiene metadatos del cliente."""
        with self._lock:
            return self._metadata.get(client_id)

    def client_exists(self, client_id: str) -> bool:
        """Verifica si el cliente existe."""
        with self._lock:
            return client_id in self._metadata

    def get_vector_store(
        self,
        client_id: str,
    ) -> Optional[FAISS]:
        """
        Obtiene el vector store del cliente desde el caché RAM.
        Retorna None si el cliente no existe o no tiene documentos cargados en caché.
        """
        if not self.client_exists(client_id):
            return None
        return self._cache.get(client_id)

    def add_documents_to_client(
        self,
        client_id: str,
        documents: List[Document],
    ) -> tuple[int, int]:
        """
        Añade documentos al cliente.
        Crea el vector store si no existe en caché, o añade al existente.
        Retorna (docs_added, chunks_added).
        """
        if not self.client_exists(client_id):
            raise ValueError(f"Cliente {client_id} no existe")

        config = self.get_client(client_id)["config"]
        embeddings = get_embeddings_for_client(config)
        if embeddings is None:
            raise ValueError("No se pudo obtener embeddings")

        chunk_size = config.get("chunk_size", 400)
        chunk_overlap = config.get("chunk_overlap", 200)
        docs_count = len(documents)

        existing_vs = self.get_vector_store(client_id)

        if existing_vs is None:
            vs = DataIngestionService.create_vector_store_from_documents(
                documents=documents,
                embeddings=embeddings,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            chunks_added = vs.index.ntotal
            self._cache.put(client_id, vs)
        else:
            chunks_added = DataIngestionService.add_documents_to_vector_store(
                vector_store=existing_vs,
                documents=documents,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

        with self._lock:
            m = self._metadata[client_id]
            stats = m["stats"]
            stats["documents_count"] = stats.get("documents_count", 0) + docs_count
            stats["chunks_count"] = stats.get("chunks_count", 0) + chunks_added
            stats["last_updated"] = datetime.utcnow()

        return docs_count, chunks_added

    def delete_client(self, client_id: str) -> None:
        """Elimina el cliente y sus datos del caché."""
        with self._lock:
            self._metadata.pop(client_id, None)
        self._cache.invalidate(client_id)

    def list_clients(self) -> List[dict]:
        """Lista metadatos de todos los clientes."""
        with self._lock:
            return list(self._metadata.values())
