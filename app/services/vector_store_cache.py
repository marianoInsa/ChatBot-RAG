import threading
from collections import OrderedDict
from typing import Optional

from langchain_community.vectorstores import FAISS

import logging

logger = logging.getLogger(__name__)


class VectorStoreCache:
    """
    Caché LRU de N índices FAISS en RAM.
    La persistencia es exclusivamente en memoria: al hacer evict, el índice se descarta.
    Los usuarios deben recargar sus documentos si el servidor se reinicia
    o si su sesión es desalojada del caché por política LRU.
    """

    def __init__(self, max_size: int):
        self._max_size = max_size
        # OrderedDict: el último accedido se mueve al final (LRU al frente)
        self._cache: OrderedDict[str, FAISS] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, client_id: str) -> Optional[FAISS]:
        """
        Obtiene el vector store del cliente desde el caché.
        Si está en caché: retorna y lo mueve al final (más reciente).
        Si no está: retorna None (sin acceso a disco ni a Blob).
        """
        with self._lock:
            if client_id in self._cache:
                self._cache.move_to_end(client_id)
                return self._cache[client_id]
            return None

    def put(self, client_id: str, vector_store: FAISS) -> None:
        """
        Inserta o actualiza el vector store en caché.
        Si el caché está lleno, evicta el elemento menos recientemente usado (LRU).
        El elemento evictado se descarta sin persistir.
        """
        with self._lock:
            self._put_unsafe(client_id, vector_store)

    def _put_unsafe(self, client_id: str, vector_store: FAISS) -> None:
        """Put sin lock. Asume que el caller ya tiene el lock."""
        while len(self._cache) >= self._max_size and self._cache:
            evict_id, _ = self._cache.popitem(last=False)
            logger.info(
                f"VectorStoreCache: evict LRU del cliente '{evict_id}' "
                f"(caché lleno, máx={self._max_size}). "
                "El usuario deberá recargar sus documentos."
            )

        self._cache[client_id] = vector_store
        self._cache.move_to_end(client_id)

    def invalidate(self, client_id: str) -> None:
        """Elimina el vector store del cliente del caché sin persistir."""
        with self._lock:
            self._cache.pop(client_id, None)

    def size(self) -> int:
        """Retorna el número de entradas actualmente en caché."""
        with self._lock:
            return len(self._cache)
