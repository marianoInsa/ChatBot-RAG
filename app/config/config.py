from pathlib import Path
import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
from functools import lru_cache

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    ollama_base_url : str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model : str = os.getenv("OLLAMA_MODEL", "llama2")
    enable_ollama : bool = os.getenv("ENABLE_OLLAMA", "false").lower() == "true"
    
    groq_model : str = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")
    google_model : str = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash-lite")

    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    user_agent: str = os.getenv("USER_AGENT", "ChatBot-RAG")
    
    chunk_size: int = 400
    chunk_overlap: int = 200
    
    hugging_face_embeddings_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    gemini_embeddings_model_name: str = "models/gemini-embedding-001"

    persist_path_huggingface: Path = BASE_DIR / "vector_store/huggingface"
    persist_path_gemini: Path = BASE_DIR / "vector_store/gemini"
    
    urls: List[str] = [
        "https://hermanos-jota-flame.vercel.app/",
        "https://hermanos-jota-flame.vercel.app/productos",
        "https://hermanos-jota-flame.vercel.app/contacto"
    ]
    file_path: Path = BASE_DIR / "corpus"

    mmr_k: int = 10 # chunks a devolver
    mmr_fetch_k: int = 50 # chunks candidatos
    mmr_lambda_mult: float = 0.5 # balance entre similarity y diversity

    max_context_length: int = 4000

    port: int = 8000

    model_config = SettingsConfigDict(
        env_file = ".env"
    )

@lru_cache()
def get_settings() -> Settings:
    return Settings()