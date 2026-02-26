"""
Fixtures compartidas para toda la suite de tests.
Estos fixtures mockean dependencias externas (Settings, APIs, etc.)
"""
import pytest
from unittest.mock import MagicMock
from pathlib import Path
from langchain_core.documents import Document


# Settings mock

@pytest.fixture
def mock_settings():
    """Retorna un objeto Settings simulado con valores de prueba."""
    settings = MagicMock()
    settings.google_api_key = "fake-google-key"
    settings.google_model = "gemini-test"
    settings.groq_api_key = "fake-groq-key"
    settings.groq_model = "groq-test"
    settings.ollama_base_url = "http://localhost:11434"
    settings.ollama_model = "llama2"
    settings.enable_ollama = False
    settings.chunk_size = 200
    settings.chunk_overlap = 50
    settings.file_path = Path("/tmp/test_corpus")
    settings.persist_path_huggingface = Path("/tmp/test_vs/huggingface")
    settings.persist_path_gemini = Path("/tmp/test_vs/gemini")
    settings.hugging_face_embeddings_model_name = "test-model"
    settings.gemini_embeddings_model_name = "test-gemini-model"
    settings.urls = ["https://example.com/page1", "https://example.com/page2"]
    settings.mmr_k = 5
    settings.mmr_fetch_k = 20
    settings.mmr_lambda_mult = 0.5
    settings.max_context_length = 500
    settings.port = 8000
    return settings


# Document helpers

@pytest.fixture
def sample_documents():
    """Lista de documentos de prueba."""
    return [
        Document(
            page_content="Contenido del documento 1",
            metadata={"source": "test.pdf", "page": 0, "title": "Test", "source_type": "pdf"}
        ),
        Document(
            page_content="Contenido del documento 2",
            metadata={"source": "test.pdf", "page": 1, "title": "Test", "source_type": "pdf"}
        ),
    ]


# Mock embeddings

@pytest.fixture
def mock_embeddings():
    """Mock de un modelo de embeddings."""
    embeddings = MagicMock()
    embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
    return embeddings


# Mock vector store

@pytest.fixture
def mock_vector_store():
    """Mock de un FAISS vector store."""
    vs = MagicMock()
    vs.index = MagicMock()
    vs.index.ntotal = 10
    vs.index.d = 384
    retriever = MagicMock()
    retriever.invoke.return_value = [
        Document(page_content="Resultado relevante", metadata={"source": "doc.pdf"})
    ]
    vs.as_retriever.return_value = retriever
    return vs


# Mock chat model

@pytest.fixture
def mock_chat_model():
    """Mock de un chat model de LangChain."""
    model = MagicMock()
    response = MagicMock()
    response.content = "Respuesta del modelo de prueba."
    model.invoke.return_value = response
    return model
