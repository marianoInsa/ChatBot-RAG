from dotenv import load_dotenv
load_dotenv()

import logging
from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from langserve import add_routes
from langchain_core.runnables import RunnableLambda

from app.embeddings.factory import get_embeddings
from app.chat_models.factory import get_chat_model

from app.services.data_service import DataIngestionService
from app.services.chat_service import ChatService

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

    chat_model = get_chat_model(chat_model="gemini")
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
rag = RunnableLambda(lambda q: chat_service.chat(q))
add_routes(
    app,
    rag,
    path="/rag",
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)