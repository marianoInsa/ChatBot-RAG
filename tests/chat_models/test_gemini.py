"""Tests para app/chat_models/gemini.py"""
from unittest.mock import patch, MagicMock
import pytest


class TestGetGemini:
    """Tests para la función get_gemini."""

    @patch("app.chat_models.gemini.ChatGoogleGenerativeAI")
    @patch("app.chat_models.gemini.settings")
    def test_con_api_key_de_usuario(self, mock_settings, mock_chat_class):
        """Debe usar la API key del usuario si se proporciona."""
        mock_settings.google_api_key = "settings-key"
        mock_settings.google_model = "gemini-test"
        mock_instance = MagicMock()
        mock_chat_class.return_value = mock_instance

        from app.chat_models.gemini import get_gemini
        result = get_gemini(user_api_key="user-key")

        mock_chat_class.assert_called_once_with(
            model="gemini-test",
            api_key="user-key"
        )
        assert result == mock_instance

    @patch("app.chat_models.gemini.ChatGoogleGenerativeAI")
    @patch("app.chat_models.gemini.settings")
    def test_con_api_key_de_settings(self, mock_settings, mock_chat_class):
        """Debe usar la API key de settings si no se proporciona user key."""
        mock_settings.google_api_key = "settings-key"
        mock_settings.google_model = "gemini-test"
        mock_instance = MagicMock()
        mock_chat_class.return_value = mock_instance

        from app.chat_models.gemini import get_gemini
        result = get_gemini(user_api_key=None)

        mock_chat_class.assert_called_once_with(
            model="gemini-test",
            api_key="settings-key"
        )
        assert result == mock_instance

    @patch("app.chat_models.gemini.settings")
    def test_sin_api_key_retorna_none(self, mock_settings):
        """Debe retornar None si no hay API key disponible."""
        mock_settings.google_api_key = ""

        from app.chat_models.gemini import get_gemini
        result = get_gemini(user_api_key=None)

        assert result is None
