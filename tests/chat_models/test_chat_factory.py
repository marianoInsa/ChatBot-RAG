"""Tests para app/chat_models/factory.py"""
from unittest.mock import patch, MagicMock
import pytest


class TestGetChatModel:
    """Tests para la función get_chat_model."""

    @patch("app.chat_models.factory.get_ollama_instance")
    @patch("app.chat_models.factory.settings")
    def test_ollama_habilitado(self, mock_settings, mock_get_ollama):
        """Debe retornar instancia Ollama cuando está habilitado."""
        mock_settings.enable_ollama = True
        mock_instance = MagicMock()
        mock_get_ollama.return_value = mock_instance

        from app.chat_models.factory import get_chat_model
        result = get_chat_model("ollama")

        mock_get_ollama.assert_called_once()
        assert result == mock_instance

    @patch("app.chat_models.factory.settings")
    def test_ollama_deshabilitado(self, mock_settings):
        """Debe retornar None cuando Ollama no está habilitado."""
        mock_settings.enable_ollama = False

        from app.chat_models.factory import get_chat_model
        result = get_chat_model("ollama")

        assert result is None

    @patch("app.chat_models.factory.get_gemini")
    @patch("app.chat_models.factory.settings")
    def test_gemini_con_key_valida(self, mock_settings, mock_get_gemini):
        """Debe retornar instancia Gemini con API key válida."""
        mock_instance = MagicMock()
        mock_get_gemini.return_value = mock_instance

        from app.chat_models.factory import get_chat_model
        result = get_chat_model("gemini", user_api_key="key")

        mock_get_gemini.assert_called_once_with("key")
        assert result == mock_instance

    @patch("app.chat_models.factory.get_gemini")
    @patch("app.chat_models.factory.settings")
    def test_gemini_sin_key(self, mock_settings, mock_get_gemini):
        """Debe retornar None si Gemini no tiene API key."""
        mock_get_gemini.return_value = None

        from app.chat_models.factory import get_chat_model
        result = get_chat_model("gemini")

        assert result is None

    @patch("app.chat_models.factory.get_groq")
    @patch("app.chat_models.factory.settings")
    def test_groq_con_key_valida(self, mock_settings, mock_get_groq):
        """Debe retornar instancia Groq con API key válida."""
        mock_instance = MagicMock()
        mock_get_groq.return_value = mock_instance

        from app.chat_models.factory import get_chat_model
        result = get_chat_model("groq", user_api_key="key")

        mock_get_groq.assert_called_once_with("key")
        assert result == mock_instance

    @patch("app.chat_models.factory.get_groq")
    @patch("app.chat_models.factory.settings")
    def test_groq_sin_key(self, mock_settings, mock_get_groq):
        """Debe retornar None si Groq no tiene API key."""
        mock_get_groq.return_value = None

        from app.chat_models.factory import get_chat_model
        result = get_chat_model("groq")

        assert result is None

    @patch("app.chat_models.factory.settings")
    def test_proveedor_desconocido(self, mock_settings):
        """Debe retornar None para un modelo no soportado."""
        from app.chat_models.factory import get_chat_model
        result = get_chat_model("modelo_inexistente")

        assert result is None
