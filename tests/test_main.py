"""Tests para app/main.py — endpoints FastAPI."""
from unittest.mock import patch, MagicMock, AsyncMock
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_app_dependencies():
    """Mock de todas las dependencias externas antes de importar app."""
    with patch("app.main.get_embeddings") as mock_emb, \
         patch("app.main.DataIngestionService") as mock_data_svc, \
         patch("app.main.get_settings") as mock_get_settings:

        mock_settings = MagicMock()
        mock_settings.enable_ollama = False
        mock_settings.port = 8000
        mock_get_settings.return_value = mock_settings

        mock_embeddings = MagicMock()
        mock_emb.return_value = mock_embeddings

        mock_vs = MagicMock()
        mock_vs.index = MagicMock()
        mock_vs.index.ntotal = 10
        mock_vs.as_retriever.return_value = MagicMock()

        mock_service = MagicMock()
        mock_service.load_vector_store.return_value = mock_vs
        mock_data_svc.return_value = mock_service

        yield {
            "embeddings": mock_embeddings,
            "vector_store": mock_vs,
            "data_service": mock_service,
            "settings": mock_settings,
        }


@pytest.fixture
def client(mock_app_dependencies):
    """TestClient de FastAPI con dependencias mockeadas."""
    from app.main import app
    with TestClient(app) as c:
        yield c


class TestHealthCheck:
    """Tests para el endpoint GET /."""

    def test_health_check_retorna_ok(self, client):
        """GET / debe retornar status ok."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "service" in data


class TestChatEndpoint:
    """Tests para el endpoint POST /api/chat."""

    @patch("app.main.ChatService")
    @patch("app.main.get_chat_model")
    def test_chat_exitoso(self, mock_get_model, mock_chat_svc_class, client):
        """Debe retornar respuesta exitosa con modelo válido."""
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        mock_chat_svc = MagicMock()
        mock_chat_svc.chat.return_value = "Respuesta de prueba."
        mock_chat_svc_class.return_value = mock_chat_svc

        response = client.post("/api/chat", json={
            "question": "¿Qué productos tienen?",
            "model_provider": "gemini",
            "api_key": "fake-key"
        })

        assert response.status_code == 200
        assert response.json()["response"] == "Respuesta de prueba."

    def test_ollama_deshabilitado(self, client, mock_app_dependencies):
        """Debe retornar 400 si se pide ollama y está deshabilitado."""
        mock_app_dependencies["settings"].enable_ollama = False

        response = client.post("/api/chat", json={
            "question": "Hola",
            "model_provider": "ollama",
            "api_key": ""
        })

        assert response.status_code == 400
        assert "Ollama" in response.json()["detail"]

    @patch("app.main.get_chat_model")
    def test_modelo_invalido(self, mock_get_model, client):
        """Debe retornar 400 si el modelo no se puede inicializar."""
        mock_get_model.return_value = None

        response = client.post("/api/chat", json={
            "question": "Hola",
            "model_provider": "gemini",
            "api_key": ""
        })

        assert response.status_code == 400


class TestRagChain:
    """Tests para la función rag_chain."""

    @patch("app.main.ChatService")
    @patch("app.main.get_chat_model")
    def test_rag_chain_con_dict(self, mock_get_model, mock_chat_svc_class, client, mock_app_dependencies):
        """rag_chain debe funcionar con input de tipo dict."""
        import app.main as main_module
        main_module.vector_store = mock_app_dependencies["vector_store"]

        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        mock_svc = MagicMock()
        mock_svc.chat.return_value = "Respuesta RAG"
        mock_chat_svc_class.return_value = mock_svc

        from app.main import rag_chain
        result = rag_chain({
            "question": "¿Horarios?",
            "model_provider": "gemini",
            "api_key": "key"
        })

        assert result.response == "Respuesta RAG"

    def test_rag_chain_sin_pregunta(self, client, mock_app_dependencies):
        """rag_chain debe lanzar ValueError si no hay pregunta."""
        import app.main as main_module
        main_module.vector_store = mock_app_dependencies["vector_store"]

        from app.main import rag_chain
        with pytest.raises(ValueError, match="No se proporcionó una pregunta"):
            rag_chain({"question": "", "model_provider": "gemini", "api_key": ""})

    def test_rag_chain_sin_vector_store(self, client):
        """rag_chain debe lanzar RuntimeError si vector_store es None."""
        import app.main as main_module
        main_module.vector_store = None

        from app.main import rag_chain
        with pytest.raises(RuntimeError, match="Vector store no está disponible"):
            rag_chain({"question": "Hola", "model_provider": "gemini", "api_key": ""})
