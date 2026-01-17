from dotenv import load_dotenv
load_dotenv()
import logging
from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
# from llms.ollama import llama2
from app.llms.groq import groq
from langchain_core.runnables import RunnableLambda
from app.services.data_service import DataIngestionService
from app.services.chat_service import ChatService

# CONFIGURACIÓN DE LOGGER
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# INICIALIZACIÓN DE SERVICIOS
data_service = DataIngestionService()
chat_service = ChatService()

try:
    vector_store = data_service.load_vector_store()
    chat_service.initialize(vector_store, groq)
    
    logger.info("Vector store y LLM cargados correctamente en ChatService.")
except Exception as e:
    logger.error(f"Error al cargar la app: {e}")
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
add_routes(
    app,
    chat_service.get_chain(),
    path="/rag",
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)