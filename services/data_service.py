from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loader import load_documents
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from uuid import uuid4

class DataIngestionService:
    def __init__(self, source: str, chunk_size: int = 1000, chunk_overlap: int = 200, embeddings_model_name: str = "sentence-transformers/all-MiniLM-L6-v2", persist_path: str = "faiss_index"):
        self.source = source
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings_model_name = embeddings_model_name
        self._embeddings = None
        self.persist_path = persist_path

    def load_data(self) -> List[Document]:
        all_docs = load_documents(self.source)
        return all_docs

    def split_corpus(self, all_docs: List[Document]) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True
        )
        all_splits = text_splitter.split_documents(all_docs)

        return all_splits
    
    # Necesito un getter para obtener una instancia de embeddings para el metodo FAISS.from_documents
    def get_embeddings_instance(self):
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.embeddings_model_name
            )
        return self._embeddings
    
    def vectorize(self) -> FAISS:
        # 1. CARGA DE DATOS
        loaded_docs = self.load_data()
        if not loaded_docs:
            raise ValueError("No se cargaron documentos.")

        # 2. SPLITTEO DE DOCS
        split_docs = self.split_corpus(loaded_docs)
        if not split_docs:
            raise ValueError("No se generaron chunks.")

        # 3. EMBEDDINGS
        embeddings = self.get_embeddings_instance()
        
        # 4. ALMACENAMIENTO
        ids = [str(uuid4()) for _ in split_docs]
        vector_store = FAISS.from_documents(
            documents=split_docs, 
            embedding=embeddings,
            ids=ids
        )
        vector_store.save_local(self.persist_path)

        return vector_store
    
    def load_vector_store(self) -> FAISS:
        embeddings = self.get_embeddings_instance()
        return FAISS.load_local(
            self.persist_path,
            embeddings
        )
    
if __name__ == "__main__":
    data_service = DataIngestionService(source="corpus")
    vector_store = data_service.vectorize()
    print("--- INFORMACION DEL PIPELINE ---")
    print(f"Fuente de datos          : {data_service.source}")
    print(f"Documentos cargados      : {len(data_service.load_data())}")
    print(f"Chunks generados         : {vector_store.index.ntotal}")
    print(f"Tamaño de chunk          : {data_service.chunk_size}")
    print(f"Overlap de chunk         : {data_service.chunk_overlap}")
    print(f"Modelo de embeddings     : {data_service.embeddings_model_name}")
    print(f"Dimensión de embeddings  : {vector_store.index.d}")