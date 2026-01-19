from dotenv import load_dotenv
load_dotenv()

import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
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

# INICIALIZACIÓN DE SERVICIOS
try:
    embeddings = get_embeddings(embeddings="huggingface")
    if embeddings is None:
        raise ValueError("Modelo de embeddings no soportado.")
    data_service = DataIngestionService(embeddings)
    vector_store = data_service.load_vector_store()

    chat_model = get_chat_model(chat_model="groq")
    if chat_model is None:
        raise ValueError("Modelo de chat no soportado.")
    chat_service = ChatService(vector_store, chat_model)
    
    logger.info("Vector store y LLM cargados correctamente en ChatService.")
except Exception as e:
    logger.error(f"Error al cargar los servicios: {e}")
    raise

app = FastAPI(
    title="Chatbot RAG Server",
    version="1.0",
    description="Chatbot RAG usando LangChain"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SERVIDOR LANGSERVE
def rag_chain(input: dict) -> ChatResponse:
    question = input["question"]
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

# INTERFAZ WEB
app.mount(
    "/static",
    StaticFiles(directory=settings.static_files_path),
    name="static"
)

@app.get("/")
def read_root():
    try:
        html_path = settings.static_files_path / "index.html"
        return FileResponse(html_path)
    except Exception as e:
        logger.error(f"Error cargando index.html: {e}")
        raise HTTPException(status_code=404, detail="Archivo no encontrado")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.port, reload=True)