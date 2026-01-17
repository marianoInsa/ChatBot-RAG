from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import FAISS
from textwrap import dedent
import logging
from app.config.config import get_settings

logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self):
        self.settings = get_settings()
        self.vector_store: FAISS | None = None
        self.retriever: BaseRetriever | None = None
        self.llm = None
        self._chain: Runnable | None = None

    def initialize(self, vector_store, llm):
        self.vector_store = vector_store
        self.llm = llm
        self.retriever = self._build_retriever()
        self._chain = self._build_chain()
        logger.info("ChatService inicializado correctamente")

    def _build_retriever(self) -> BaseRetriever:
        if not self.vector_store:
            raise RuntimeError("Vector store no inicializado correctamente")
        
        return self.vector_store.as_retriever(
            search_type="mmr", # Maximal Marginal Relevance
            search_kwargs={
                "k": self.settings.mmr_k, # numero de documentos a devolver
                "fetch_k": self.settings.mmr_fetch_k, # numero de documentos a considerar para MMR
                "lambda_mult": self.settings.mmr_lambda_mult # 0.5 balancea entre similarity vs diversity
            }
        )
    
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

        Pregunta del cliente: {input}
        """)
        prompt = ChatPromptTemplate.from_template(system_template)
        return prompt
    
    @staticmethod
    def format_docs(docs) -> str:
        return "\n\n".join(
            doc.page_content for doc in docs if doc.page_content
        )
    
    def _build_chain(self) -> Runnable:
        if not self.retriever or not self.llm:
            raise RuntimeError("ChatService no inicializado correctamente")

        logger.info("Construyendo cadena RAG...")

        return (
            RunnableParallel({
                "context": self.retriever | self.format_docs, 
                "input": RunnablePassthrough()
            })
            | self._build_prompt()
            | self.llm
            | StrOutputParser()
        )
    
    def get_chain(self) -> Runnable:
        if self._chain is None:
            raise RuntimeError("ChatService no inicializado correctamente")

        return self._chain

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    from app.llms.groq import groq
    from app.services.data_service import DataIngestionService

    data_service = DataIngestionService()
    vector_store = data_service.load_vector_store()

    chat_service = ChatService(
        vector_store=vector_store, 
        llm=groq
    )
    query = "Hola, que productos venden?"

    response = chat_service.get_chain().invoke(query)
    print(response)