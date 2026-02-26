"""Tests para app/embedding_models/factory.py"""
from unittest.mock import patch, MagicMock
import pytest


class TestGetEmbeddings:
    """Tests para la función get_embeddings."""

    @patch("app.embedding_models.factory.hugging_face_embeddings")
    def test_default_retorna_huggingface(self, mock_hf):
        """Debe retornar embeddings de HuggingFace por defecto."""
        from app.embedding_models.factory import get_embeddings
        result = get_embeddings("default")

        assert result == mock_hf

    @patch("app.embedding_models.factory.get_gemini_embeddings")
    def test_gemini_retorna_gemini_embeddings(self, mock_get_gemini):
        """Debe retornar embeddings de Gemini cuando se solicita."""
        mock_instance = MagicMock()
        mock_get_gemini.return_value = mock_instance

        from app.embedding_models.factory import get_embeddings
        result = get_embeddings("gemini")

        mock_get_gemini.assert_called_once()
        assert result == mock_instance

    def test_modelo_no_soportado_retorna_none(self):
        """Debe retornar None para un modelo de embeddings no soportado."""
        from app.embedding_models.factory import get_embeddings
        result = get_embeddings("modelo_inexistente")

        assert result is None
