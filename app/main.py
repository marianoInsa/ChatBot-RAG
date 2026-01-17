from dotenv import load_dotenv
load_dotenv()
import logging
from fastapi import FastAPI
import uvicorn
# from contextlib import asynccontextmanager
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
# from llms.ollama import llama2
from app.llms.groq import groq

from app.services.data_service import DataIngestionService

# CONFIGURACIÓN DE LOGGER
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # CARGA DEL VECTOR STORE

#     yield
#     # LIMPIEZA FINAL

app = FastAPI(
    title="Chatbot RAG Server",
    version="1.0",
    description="Chatbot RAG usando LangChain",
    # lifespan=lifespan
)

data_service = DataIngestionService()
vector_store = data_service.load_vector_store()

# 3. CONFIG DEL RETRIEVER
retriever = vector_store.as_retriever(
    search_type="mmr", # Maximal Marginal Relevance
    search_kwargs={
        "k": 5, # numero de documentos a devolver
        "fetch_k": 20, # numero de documentos a considerar para MMR
        "lambda_mult": 0.5 # 0.5 balancea entre similarity vs diversity
    }
)

# 4. MODELO LLM (Ollama)


# 5. RAG CHAIN
system_template = """Eres un Asistente Virtual experto de la mueblería "Hermanos Jota". 
Tu trabajo es responder en español a preguntas sobre la mueblería de manera profesional y comercial.

Instrucciones:
1. Usa SOLO el contexto proporcionado abajo para responder. No inventes información.
2. Si la respuesta no está en el contexto, di amablemente que no tienes esa información y sugiere contactar a ventas.
3. Sé breve y conciso. Evita introducciones largas como "Basado en el contexto...". Ve al grano.
4. Mantén un tono cordial y servicial.

<context>
{context}
</context>

Pregunta del cliente: {input}
"""
prompt = ChatPromptTemplate.from_template(system_template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    RunnableParallel(
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
    )
    | prompt
    | groq
    | StrOutputParser()
)

print("Cadena RAG lista.")

# 6. SERVIDOR
add_routes(
    app, 
    rag_chain, 
    path="/rag"
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)