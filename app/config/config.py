from pathlib import Path
import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
from functools import lru_cache

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    user_agent: str = os.getenv("USER_AGENT", "ChatBot-RAG")
    
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    hugging_face_embeddings_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    gemini_embeddings_model_name: str = "models/gemini-embedding-001"

    persist_path: Path = BASE_DIR / "faiss_index"
    urls: List[str] = [
        "https://hermanos-jota-flame.vercel.app/",
        "https://hermanos-jota-flame.vercel.app/productos",
        "https://hermanos-jota-flame.vercel.app/contacto"
    ]
    file_path: Path = BASE_DIR / "corpus"

    mmr_k: int = 5
    mmr_fetch_k: int = 20
    mmr_lambda_mult: float = 0.5

    max_context_length: int = 4000

    model_config = SettingsConfigDict(
        env_file = ".env"
    )

@lru_cache()
def get_settings() -> Settings:
    return Settings()