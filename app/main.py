from dotenv import load_dotenv
load_dotenv()

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from langserve import add_routes
from langchain_core.runnables import RunnableLambda
from app.models.chat_models import ChatQuestion, ChatResponse

from app.embedding_models.factory import get_embeddings
from app.chat_models.factory import get_chat_model

from app.services.data_service import DataIngestionService
from app.services.chat_service import ChatService

from app.config.config import get_settings
settings = get_settings()

# CONFIGURACIÓN DE LOGGER
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# variable global para el vector store
vector_store = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # INICIALIZACIÓN DE SERVICIOS
    global vector_store
    try:
        embeddings = get_embeddings()
        if embeddings is None:
            raise ValueError("Modelo de embeddings no soportado.")
        data_service = DataIngestionService(embeddings)
        vector_store = data_service.load_vector_store()

        logger.info("Vector store cargado correctamente.")
    except Exception as e:
        logger.error(f"Error al cargar los servicios: {e}")
        raise
    
    yield

app = FastAPI(
    title="Chatbot RAG Server",
    version="1.0",
    description="Chatbot RAG usando LangChain",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=JSONResponse)
def health_check():
    return {"status": "ok", "service": "ChatBot RAG API"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatQuestion):
    if request.model_provider == "ollama" and not settings.enable_ollama:
        raise HTTPException(
            status_code=400,
            detail="Ollama sólo está habilitado localmente."
        )

    try:
        chat_model = get_chat_model(
            chat_model=request.model_provider,
            user_api_key=request.api_key
        )
        if chat_model is None:
            raise HTTPException(
                status_code=400,
                detail=f"No se pudo iniciar {request.model_provider}. Verifique su API Key."
            )

        chat_service = ChatService(vector_store, chat_model)

        response = chat_service.chat(request.question)

        return ChatResponse(response=response)

    except HTTPException as http_exc:
        raise http_exc

    except Exception as e:
        logger.error(f"Error en el endpoint de chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# SERVIDOR LANGSERVE
def rag_chain(request: ChatQuestion | dict) -> ChatResponse:
    if vector_store is None:
        raise RuntimeError("Vector store no está disponible. Espera a que la aplicación inicie.")

    if isinstance(request, dict):
        question = request.get("question", "")
        model_provider = request.get("model_provider", "ollama")
        api_key = request.get("api_key", "")
    else:
        question = request.question
        model_provider = request.model_provider
        api_key = request.api_key

    if not question:
        raise ValueError("No se proporcionó una pregunta.")

    chat_model = get_chat_model(
        chat_model=model_provider,
        user_api_key=api_key
    )
    if chat_model is None:
        raise ValueError("Modelo de chat no soportado.")

    chat_service = ChatService(vector_store, chat_model)
    response = chat_service.chat(question)
    return ChatResponse(response=response)

rag = RunnableLambda(rag_chain).with_types(
    input_type=ChatQuestion,
    output_type=ChatResponse
)

add_routes(
    app,
    rag,
    path="/rag",
)

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.port, reload=True)