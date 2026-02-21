from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional, List, Literal


class ClientConfig(BaseModel):
    """Configuración opcional por cliente. Usa defaults globales si no se especifica."""

    embedding_provider: Optional[Literal["huggingface", "gemini"]] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    mmr_k: Optional[int] = None
    mmr_fetch_k: Optional[int] = None
    mmr_lambda_mult: Optional[float] = None


class ClientRegisterRequest(BaseModel):
    """Request para registro de nuevo cliente."""

    name: Optional[str] = None
    config: Optional[ClientConfig] = None


class ClientRegisterResponse(BaseModel):
    """Respuesta del registro de cliente."""

    client_id: str
    created_at: datetime
    message: str = "Cliente registrado correctamente. Usa el client_id para cargar documentos y chatear."


class ClientStats(BaseModel):
    """Estadísticas del cliente."""

    documents_count: int = 0
    chunks_count: int = 0
    last_updated: Optional[datetime] = None


class ClientInfoResponse(BaseModel):
    """Información completa del cliente."""

    client_id: str
    created_at: datetime
    config: dict
    stats: ClientStats


class DocumentUploadResponse(BaseModel):
    """Respuesta de la carga de documentos."""

    success: bool
    pdfs_processed: int = 0
    urls_processed: int = 0
    total_chunks_added: int = 0
    message: str = ""
    errors: Optional[List[str]] = None
