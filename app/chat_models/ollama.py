from langchain_ollama import ChatOllama
from app.config.config import get_settings
settings = get_settings()

def get_ollama_instance():
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        validate_model_on_init=True,
        temperature=0.2, # subirlo lo hace mas creativo
        # seed=77, # setear una seed hace que las respuestas sean reproducibles
        num_predict=256
    )