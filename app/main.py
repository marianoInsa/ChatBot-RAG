from dotenv import load_dotenv

load_dotenv()

import json
import logging
import tempfile
import uuid
from pathlib import Path

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from app.models.chat_models import ChatQuestion, ChatResponse
from app.models.client_models import (
    ClientInfoResponse,
    ClientRegisterRequest,
    ClientRegisterResponse,
    ClientStats,
    DocumentUploadResponse,
)

from app.embedding_models.factory import get_embeddings
from app.chat_models.factory import get_chat_model
from app.config.config import get_settings
from app.loaders.loader import load_documents_from_sources
from app.services.chat_service import ChatService
from app.services.client_manager import ClientManager
from app.services.vector_store_cache import VectorStoreCache

# CONFIGURACIÓN DE LOGGER
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()

# Servicios globales (inicializados en lifespan)
client_manager: ClientManager | None = None

# Límites
MAX_FILE_SIZE_MB = 50
MAX_FILES = 10
MAX_URLS = 20


def _validate_client_id(client_id: str) -> None:
    try:
        uuid.UUID(client_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="client_id inválido (debe ser UUID)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client_manager
    try:
        cache = VectorStoreCache(max_size=settings.vector_store_cache_size)
        client_manager = ClientManager(cache=cache)
        logger.info(
            f"ClientManager inicializado. Caché LRU en RAM (máx {settings.vector_store_cache_size} índices FAISS)."
        )
    except Exception as e:
        logger.error(f"Error al inicializar servicios: {e}")
        raise

    yield


app = FastAPI(
    title="Chatbot RAG API Multi-Tenant",
    version="2.0",
    description="API genérica ChatBot RAG multi-tenant. Registra clientes, carga documentos (PDFs/URLs) y chatea.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# INTERFAZ WEB
app.mount(
    "/static",
    StaticFiles(directory=settings.static_files_path),
    name="static",
)


@app.get("/")
def read_root():
    try:
        html_path = settings.static_files_path / "index.html"
        return FileResponse(html_path)
    except Exception as e:
        logger.error(f"Error cargando index.html: {e}")
        raise HTTPException(status_code=404, detail="Archivo no encontrado")


# ----- REGISTRO -----
@app.post("/api/clients/register", response_model=ClientRegisterResponse)
async def register_client(request: ClientRegisterRequest):
    if client_manager is None:
        raise HTTPException(status_code=503, detail="Servicio no inicializado")
    client_id = client_manager.register_client(config=request.config)
    return ClientRegisterResponse(
        client_id=client_id,
        created_at=client_manager.get_client(client_id)["created_at"],
    )


# ----- CARGA DE DOCUMENTOS -----
@app.post("/api/clients/{client_id}/documents/upload", response_model=DocumentUploadResponse)
async def upload_documents(
    client_id: str,
    files: list[UploadFile] | None = File(default=None),
    urls: str = Form(default="[]"),
):
    _validate_client_id(client_id)
    if client_manager is None:
        raise HTTPException(status_code=503, detail="Servicio no inicializado")
    if not client_manager.client_exists(client_id):
        raise HTTPException(status_code=404, detail="Cliente no encontrado")

    pdf_paths: list[Path] = []
    urls_list: list[str] = []
    files = files or []

    errors: list[str] = []

    # Procesar PDFs
    if len(files) > MAX_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Máximo {MAX_FILES} archivos por request",
        )
    for f in files:
        if f.filename and f.filename.lower().endswith(".pdf"):
            content = await f.read()
            if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
                errors.append(f"{f.filename}: supera {MAX_FILE_SIZE_MB}MB")
                continue
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(content)
                pdf_paths.append(Path(tmp.name))
        else:
            errors.append(f"{f.filename or 'archivo'}: solo se aceptan PDFs")

    # Procesar URLs
    try:
        parsed = json.loads(urls) if isinstance(urls, str) else urls
        urls_list = [u for u in (parsed if isinstance(parsed, list) else []) if isinstance(u, str)]
    except json.JSONDecodeError:
        pass
    if len(urls_list) > MAX_URLS:
        urls_list = urls_list[:MAX_URLS]

    if not pdf_paths and not urls_list:
        return DocumentUploadResponse(
            success=False,
            message="No se proporcionaron PDFs ni URLs válidos.",
            errors=errors if errors else None,
        )

    try:
        documents = load_documents_from_sources(pdf_paths=pdf_paths, urls=urls_list)
    except Exception as e:
        logger.error(f"Error cargando documentos: {e}")
        return DocumentUploadResponse(
            success=False,
            message=str(e),
            errors=errors,
        )
    finally:
        for p in pdf_paths:
            try:
                p.unlink()
            except OSError:
                pass

    if not documents:
        return DocumentUploadResponse(
            success=False,
            message="No se extrajo contenido de los documentos.",
            pdfs_processed=len(pdf_paths),
            urls_processed=len(urls_list),
            errors=errors if errors else None,
        )

    try:
        docs_added, chunks_added = client_manager.add_documents_to_client(client_id, documents)
        return DocumentUploadResponse(
            success=True,
            pdfs_processed=len(pdf_paths),
            urls_processed=len(urls_list),
            total_chunks_added=chunks_added,
            message=f"Documentos cargados: {docs_added}, chunks añadidos: {chunks_added}.",
            errors=errors if errors else None,
        )
    except Exception as e:
        logger.error(f"Error añadiendo documentos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ----- CHAT -----
@app.post("/api/clients/{client_id}/chat", response_model=ChatResponse)
async def chat_endpoint(client_id: str, request: ChatQuestion):
    _validate_client_id(client_id)
    if client_manager is None:
        raise HTTPException(status_code=503, detail="Servicio no inicializado")
    if not client_manager.client_exists(client_id):
        raise HTTPException(status_code=404, detail="Cliente no encontrado")

    vector_store = client_manager.get_vector_store(client_id)
    if vector_store is None:
        raise HTTPException(
            status_code=400,
            detail="El cliente no tiene documentos cargados. Carga PDFs o URLs primero.",
        )

    if request.model_provider == "ollama" and not settings.enable_ollama:
        raise HTTPException(
            status_code=400,
            detail="Ollama sólo está habilitado localmente.",
        )

    chat_model = get_chat_model(
        chat_model=request.model_provider,
        user_api_key=request.api_key,
    )
    if chat_model is None:
        raise HTTPException(
            status_code=400,
            detail=f"No se pudo iniciar {request.model_provider}. Verifique su API Key.",
        )

    chat_service = ChatService(vector_store, chat_model)
    response = chat_service.chat(request.question)
    return ChatResponse(response=response)


# ----- INFO CLIENTE -----
@app.get("/api/clients/{client_id}", response_model=ClientInfoResponse)
async def get_client_info(client_id: str):
    _validate_client_id(client_id)
    if client_manager is None:
        raise HTTPException(status_code=503, detail="Servicio no inicializado")
    client = client_manager.get_client(client_id)
    if client is None:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
    stats = client["stats"]
    return ClientInfoResponse(
        client_id=client["client_id"],
        created_at=client["created_at"],
        config=client["config"],
        stats=ClientStats(
            documents_count=stats.get("documents_count", 0),
            chunks_count=stats.get("chunks_count", 0),
            last_updated=stats.get("last_updated"),
        ),
    )


# ----- ADMIN -----
@app.get("/api/admin/clients")
async def list_clients_admin():
    if client_manager is None:
        raise HTTPException(status_code=503, detail="Servicio no inicializado")
    return client_manager.list_clients()


@app.delete("/api/admin/clients/{client_id}")
async def delete_client_admin(client_id: str):
    _validate_client_id(client_id)
    if client_manager is None:
        raise HTTPException(status_code=503, detail="Servicio no inicializado")
    if not client_manager.client_exists(client_id):
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
    client_manager.delete_client(client_id)
    return {"message": "Cliente eliminado"}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.port, reload=True)
