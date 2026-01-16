from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from services.loader import DocumentLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from uuid import uuid4
import logging
from services.config import get_settings

logger = logging.getLogger(__name__)

class DataIngestionService:
    def __init__(self):
        self.settings = get_settings()
        self._embeddings = None

    def load_data(self) -> List[Document]:
        doc_loader = DocumentLoader()
        all_docs = doc_loader.load_documents(self.settings.file_path)
        return all_docs

    def split_corpus(self, all_docs: List[Document]) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            add_start_index=True
        )
        all_splits = text_splitter.split_documents(all_docs)

        return all_splits
    
    # Necesito un getter para obtener una instancia de embeddings para el metodo FAISS.from_documents
    def get_embeddings_instance(self):
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.settings.embeddings_model_name
            )
        return self._embeddings
    
    def vectorize(self) -> FAISS:
        # 1. CARGA DE DATOS
        loaded_docs = self.load_data()
        if not loaded_docs:
            logger.warning("No se cargaron documentos.")
            raise
        logger.info(f"Documentos cargados: {len(loaded_docs)}")

        # 2. SPLITTEO DE DOCS
        split_docs = self.split_corpus(loaded_docs)
        if not split_docs:
            logger.warning("No se generaron chunks.")
            raise
        logger.info(f"Chunks generados: {len(split_docs)}")

        # 3. EMBEDDINGS
        embeddings = self.get_embeddings_instance()
        logger.info(f"Usando modelo de embeddings: {self.settings.embeddings_model_name}")
        
        # 4. ALMACENAMIENTO
        ids = [str(uuid4()) for _ in split_docs]
        vector_store = FAISS.from_documents(
            documents=split_docs, 
            embedding=embeddings,
            ids=ids
        )
        vector_store.save_local(self.settings.persist_path)
        logger.info(f"Vector store guardado en: {self.settings.persist_path}")
        if not vector_store or vector_store.index.ntotal == 0:
            logger.warning("El vector store no se cargó correctamente.")
            raise

        return vector_store
    
    def load_vector_store(self) -> FAISS:
        embeddings = self.get_embeddings_instance()
        return FAISS.load_local(
            self.settings.persist_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    
if __name__ == "__main__":
    data_service = DataIngestionService()
    vector_store = data_service.vectorize()
    print("--- INFORMACION DEL PIPELINE ---")
    print(f"Fuente de datos          : {data_service.settings.file_path}")
    print(f"Documentos cargados      : {len(data_service.load_data())}")
    print(f"Chunks generados         : {vector_store.index.ntotal}")
    print(f"Tamaño de chunk          : {data_service.settings.chunk_size}")
    print(f"Overlap de chunk         : {data_service.settings.chunk_overlap}")
    print(f"Modelo de embeddings     : {data_service.settings.embeddings_model_name}")
    print(f"Dimensión de embeddings  : {vector_store.index.d}")