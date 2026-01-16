import os
from pydantic_settings import BaseSettings
from typing import List
from functools import lru_cache

class Settings(BaseSettings):
    user_agent: str = os.getenv("USER_AGENT", "ChatBot-RAG")
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embeddings_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    persist_path: str = "faiss_index"
    urls: List[str] = [
        "https://hermanos-jota-flame.vercel.app/",
        "https://hermanos-jota-flame.vercel.app/productos",
        "https://hermanos-jota-flame.vercel.app/contacto"
    ]
    file_path: str = "corpus"

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings()