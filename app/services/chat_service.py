from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.vectorstores import FAISS
from textwrap import dedent
import logging
from app.config.config import get_settings

logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self, vector_store: FAISS, chat_model: BaseChatModel):
        self.settings = get_settings()
        self.vector_store: FAISS = vector_store
        self.chat_model: BaseChatModel = chat_model
        self.retriever: BaseRetriever = self._build_retriever()
        self.prompt: ChatPromptTemplate = self._build_prompt()

    def _build_retriever(self) -> BaseRetriever:
        return self.vector_store.as_retriever(
            search_type="mmr", # Maximal Marginal Relevance
            search_kwargs={
                "k": self.settings.mmr_k, # numero de documentos a devolver
                "fetch_k": self.settings.mmr_fetch_k, # numero de documentos a considerar para MMR
                "lambda_mult": self.settings.mmr_lambda_mult # 0.5 balancea entre similarity vs diversity
            }
        )
    
    def retrieve(self, query: str) -> List[Document]:
        logger.info("Recuperando documentos...")
        relevant_docs = self.retriever.invoke(query)
        return relevant_docs
    
    def _build_prompt(self) -> ChatPromptTemplate:
        system_template = dedent("""
        Eres un Asistente Virtual experto de la mueblería "Hermanos Jota".
        Tu trabajo es responder en español a preguntas sobre la mueblería de manera profesional y comercial.

        Instrucciones:
        1. Usa SOLO el contexto proporcionado abajo para responder. No inventes información.
        2. Si la respuesta no está en el contexto, di amablemente que no tienes esa información y sugiere contactar a ventas.
        3. Sé breve y conciso. Evita introducciones largas como "Basado en el contexto...". Ve al grano.
        4. Mantén un tono cordial y servicial.

        <context>
        {context}
        </context>

        Pregunta del cliente: {question}
        """)
        
        return ChatPromptTemplate.from_template(system_template)
    
    @staticmethod
    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join(
            doc.page_content for doc in docs if doc.page_content
        )
    
    def generate_response(self, question: str, context: str) -> str:
        messages = self.prompt.format_messages(
            context=context,
            question=question
        )
        response = self.chat_model.invoke(messages)
        return response.content
    
    def chat(self, question: str) -> str:
        logger.info("Procesando la pregunta...")
        
        relevant_docs = self.retrieve(question)
        if not relevant_docs:
            return "Lo siento, no tengo información disponible para responder a su pregunta."
        
        context = self.format_docs(relevant_docs)
        if len(context) > self.settings.max_context_length:
            context = context[: self.settings.max_context_length]
            logger.info("Contexto truncado a la longitud máxima.")

        response = self.generate_response(question, context)
        return response

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    from app.chat_models.groq import get_groq
    from app.chat_models.gemini import get_gemini
    from app.services.data_service import DataIngestionService
    from app.embeddings.huggingface import hugging_face_embeddings

    data_service = DataIngestionService(hugging_face_embeddings)
    vector_store = data_service.load_vector_store()

    chat_service = ChatService(
        vector_store=vector_store, 
        chat_model=get_groq()
    )
    query = "Hola, que productos venden?"

    response = chat_service.chat(query)
    print(response)