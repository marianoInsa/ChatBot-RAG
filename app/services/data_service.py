from typing import List
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.loaders.loader import load_documents
from langchain_community.vectorstores import FAISS
from uuid import uuid4
import logging
from app.config.config import get_settings

logger = logging.getLogger(__name__)

class DataIngestionService:
    def __init__(self, embeddings: Embeddings):
        self.settings = get_settings()
        self._embeddings: Embeddings = embeddings
        self._persist_path = self._set_persist_path()
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", " ", ""]
        )
        self._vector_store = None

    def _set_persist_path(self):
        if isinstance(self._embeddings, HuggingFaceEmbeddings):
            return self.settings.persist_path_huggingface
        elif isinstance(self._embeddings, GoogleGenerativeAIEmbeddings):
            return self.settings.persist_path_gemini
        else:
            logger.error("Tipo de embeddings no soportado para persistencia.")
            raise

    def load_and_chunk(self) -> List[Document]:
        all_docs = load_documents(self.settings.file_path)
        chunks = self._text_splitter.split_documents(all_docs)
        return all_docs, chunks

    def vectorize(self) -> FAISS:
        if self._persist_path.exists():
            logger.info("Eliminando vector store existente para recrearlo...")
            
            # si ya existe se elimina para recrearlo
            for file in self._persist_path.iterdir():
                file.unlink()
            self._persist_path.rmdir()
            
            logger.info("Vector store eliminado. Recreando...")
        
        # CARGA DE DATOS Y CHUNKING
        loaded_docs, chunks = self.load_and_chunk()
        if not loaded_docs:
            logger.warning("No se cargaron documentos.")
            raise
        logger.info(f"Documentos cargados: {len(loaded_docs)}")
        logger.info(f"Chunks generados: {len(chunks)}")
        
        # ALMACENAMIENTO
        ids = [str(uuid4()) for _ in chunks]
        self._vector_store = FAISS.from_documents(
            documents=chunks, 
            embedding=self._embeddings,
            ids=ids
        )
        
        # PERSISTENCIA
        self._vector_store.save_local(self._persist_path)
        
        self.print_vector_store_info()

        if not self._vector_store or self._vector_store.index.ntotal == 0:
            logger.warning("El vector store no se cargó correctamente.")
            raise

        return self._vector_store

    def print_vector_store_info(self):
        if self._vector_store is None:
            logger.warning("El vector store no ha sido inicializado.")
            return
        logger.info("--- INFORMACIÓN DEL VECTOR STORE ---")
        logger.info(f"Modelo de Embeddings:\t{type(self._embeddings).__name__}")
        logger.info(f"Almacenamiento:\t{self._persist_path}")
        logger.info(f"Índice:\t\t{self._vector_store.index.ntotal}")
        logger.info(f"Dimensión:\t{self._vector_store.index.d}")
        logger.info("-------------------------------------")

    @staticmethod
    def create_text_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
        """Crea un text splitter con la configuración dada."""
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", " ", ""]
        )

    @staticmethod
    def add_documents_to_vector_store(
        vector_store: FAISS,
        documents: List[Document],
        chunk_size: int,
        chunk_overlap: int,
    ) -> int:
        """
        Añade documentos a un vector store existente.
        Retorna el número de chunks añadidos.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        if not chunks:
            return 0
        ids = [str(uuid4()) for _ in chunks]
        texts = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        vector_store.add_texts(texts, metadatas=metadatas, ids=ids)
        return len(chunks)

    @staticmethod
    def create_vector_store_from_documents(
        documents: List[Document],
        embeddings: Embeddings,
        chunk_size: int,
        chunk_overlap: int,
    ) -> FAISS:
        """
        Crea un nuevo vector store FAISS desde documentos.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        if not chunks:
            raise ValueError("No se generaron chunks. Verifica que los documentos tengan contenido.")
        ids = [str(uuid4()) for _ in chunks]
        return FAISS.from_documents(
            documents=chunks,
            embedding=embeddings,
            ids=ids
        )

    def load_vector_store(self) -> FAISS:
        if self._vector_store is not None:
            return self._vector_store

        if self._persist_path.exists():
            logger.info(f"Cargando vector store desde: {self._persist_path}")
            self._vector_store = FAISS.load_local(
                self._persist_path,
                self._embeddings,
                allow_dangerous_deserialization=True
            )
            self.print_vector_store_info()
            return self._vector_store
        
        logger.info("El vector store aún no ha sido creado. Iniciando vectorización...")
        return self.vectorize()
    
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    # from app.embeddings.gemini import get_gemini_embeddings
    from app.embedding_models.huggingface import hugging_face_embeddings
    # embeddings = get_gemini_embeddings()
    data_service = DataIngestionService(embeddings=hugging_face_embeddings)
    vector_store = data_service.vectorize()
    print("--- INFORMACION DEL PIPELINE ---")
    print(f"Fuente de datos          : {data_service.settings.file_path}")
    print(f"Documentos cargados      : {len(data_service.load_and_chunk()[0])}")
    print(f"Chunks generados         : {vector_store.index.ntotal}")
    print(f"Tamaño de chunk          : {data_service.settings.chunk_size}")
    print(f"Overlap de chunk         : {data_service.settings.chunk_overlap}")
    print(f"Modelo de embeddings     : {data_service._embeddings}")
    print(f"Dimensión de embeddings  : {vector_store.index.d}")