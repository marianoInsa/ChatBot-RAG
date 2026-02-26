"""Tests para app/chat_models/groq.py"""
from unittest.mock import patch, MagicMock
import pytest


class TestGetGroq:
    """Tests para la función get_groq."""

    @patch("app.chat_models.groq.ChatGroq")
    @patch("app.chat_models.groq.settings")
    def test_con_api_key_de_usuario(self, mock_settings, mock_chat_class):
        """Debe usar la API key del usuario si se proporciona."""
        mock_settings.groq_api_key = "settings-key"
        mock_settings.groq_model = "groq-test"
        mock_instance = MagicMock()
        mock_chat_class.return_value = mock_instance

        from app.chat_models.groq import get_groq
        result = get_groq(user_api_key="user-key")

        mock_chat_class.assert_called_once_with(
            api_key="user-key",
            model="groq-test",
            temperature=0.1,
            max_tokens=1024
        )
        assert result == mock_instance

    @patch("app.chat_models.groq.ChatGroq")
    @patch("app.chat_models.groq.settings")
    def test_con_api_key_de_settings(self, mock_settings, mock_chat_class):
        """Debe usar la API key de settings si no se proporciona user key."""
        mock_settings.groq_api_key = "settings-key"
        mock_settings.groq_model = "groq-test"
        mock_instance = MagicMock()
        mock_chat_class.return_value = mock_instance

        from app.chat_models.groq import get_groq
        result = get_groq(user_api_key=None)

        mock_chat_class.assert_called_once_with(
            api_key="settings-key",
            model="groq-test",
            temperature=0.1,
            max_tokens=1024
        )
        assert result == mock_instance

    @patch("app.chat_models.groq.settings")
    def test_sin_api_key_retorna_none(self, mock_settings):
        """Debe retornar None si no hay API key disponible."""
        mock_settings.groq_api_key = ""

        from app.chat_models.groq import get_groq
        result = get_groq(user_api_key=None)

        assert result is None
