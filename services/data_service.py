from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loader import load_documents
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class DataIngestionService:
    def __init__(self, source: str, chunk_size: int = 1000, chunk_overlap: int = 200, embeddings_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.source = source
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings_model_name = embeddings_model_name

    def load_data(self):
        all_docs = load_documents(self.source)
        return all_docs

    def split_corpus(self, all_docs: List) -> List:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True
        )
        all_splits = text_splitter.split_documents(all_docs)

        return all_splits
    # Necesito un getter para obtener una instancia de embeddings para el metodo FAISS.from_documents
    def get_embeddings_instance(self):
        return HuggingFaceEmbeddings(model_name=self.embeddings_model_name)
    
    def vectorize(self) -> List:
        # 1. CARGA DE DATOS
        loaded_docs = self.load_data()

        # 2. SPLITTEO DE DOCS
        split_docs = self.split_corpus(loaded_docs)
        if not split_docs:
          raise ValueError(
              "No se generaron chunks. Revisar PDFs (pueden estar vacíos o escaneados)."
          )

        # 3. EMBEDDINGS
        embeddings = self.get_embeddings_instance()
        
        # 4. ALMACENAMIENTO
        vector_store = FAISS.from_documents(documents=split_docs, embedding=embeddings)

        return vector_store
    
if __name__ == "__main__":
    data_service = DataIngestionService(source="corpus")
    vector_store = data_service.vectorize()
    print("--- INFORMACION DEL PIPELINE ---")
    print(f"Documentos vectorizados: {vector_store.index.ntotal}")
    print(f"Dimensiones de los vectores: {vector_store.index.d} ")