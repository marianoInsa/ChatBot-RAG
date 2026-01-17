from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.loaders.loader import load_documents
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from uuid import uuid4
import logging
from app.config.config import get_settings

logger = logging.getLogger(__name__)

class DataIngestionService:
    def __init__(self):
        self.settings = get_settings()
        self._embeddings = HuggingFaceEmbeddings(
                model_name=self.settings.embeddings_model_name
            )
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", " ", ""]
        )
        self._vector_store = None

    def load_and_chunk(self) -> List[Document]:
        all_docs = load_documents(self.settings.file_path)
        all_chunks = self._text_splitter.split_documents(all_docs)
        return all_chunks

    def vectorize(self) -> FAISS:
        # CARGA DE DATOS Y CHUNKING
        loaded_docs = self.load_and_chunk()
        if not loaded_docs:
            logger.warning("No se cargaron documentos.")
            raise
        logger.info(f"Documentos cargados: {len(loaded_docs)}")
        
        # ALMACENAMIENTO
        ids = [str(uuid4()) for _ in loaded_docs]
        self._vector_store = FAISS.from_documents(
            documents=loaded_docs, 
            embedding=self._embeddings,
            ids=ids
        )
        self._vector_store.save_local(self.settings.persist_path)
        logger.info("--- VECTOR STORE ---")
        logger.info(f"Guardado en: {self.settings.persist_path}")
        logger.info(f"Chunks generados (FAISS index): {self._vector_store.index.ntotal}")
        logger.info(f"Dimensión de embeddings: {self._vector_store.index.d}")
        logger.info("--------------------")

        if not self._vector_store or self._vector_store.index.ntotal == 0:
            logger.warning("El vector store no se cargó correctamente.")
            raise

        return self._vector_store

    def load_vector_store(self) -> FAISS:
        if self._vector_store is not None:
            return self._vector_store
        
        if self.settings.persist_path.exists():
            logger.info(f"Cargando vector store desde: {self.settings.persist_path}")
            self._vector_store = FAISS.load_local(
                self.settings.persist_path,
                self._embeddings,
                allow_dangerous_deserialization=True
            )
            return self._vector_store
        
        logger.info("El vector store aún no ha sido creado. Iniciando vectorización...")
        return self.vectorize()
    
if __name__ == "__main__":
    data_service = DataIngestionService()
    vector_store = data_service.vectorize()
    print("--- INFORMACION DEL PIPELINE ---")
    print(f"Fuente de datos          : {data_service.settings.file_path}")
    # print(f"Documentos cargados      : {len(data_service.load_data())}")
    print(f"Chunks generados         : {vector_store.index.ntotal}")
    print(f"Tamaño de chunk          : {data_service.settings.chunk_size}")
    print(f"Overlap de chunk         : {data_service.settings.chunk_overlap}")
    print(f"Modelo de embeddings     : {data_service.settings.embeddings_model_name}")
    print(f"Dimensión de embeddings  : {vector_store.index.d}")