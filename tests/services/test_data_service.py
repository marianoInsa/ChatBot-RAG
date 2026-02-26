"""Tests para app/services/data_service.py"""
from unittest.mock import patch, MagicMock, PropertyMock
from pathlib import Path
from langchain_core.documents import Document
import pytest


class TestDataIngestionService:
    """Tests para la clase DataIngestionService."""

    @patch("app.services.data_service.get_settings")
    def test_persist_path_huggingface(self, mock_get_settings):
        """Debe asignar el path de persistencia correcto para HuggingFace."""
        from langchain_huggingface import HuggingFaceEmbeddings

        mock_settings = MagicMock()
        mock_settings.persist_path_huggingface = Path("/tmp/vs/hf")
        mock_settings.persist_path_gemini = Path("/tmp/vs/gemini")
        mock_settings.chunk_size = 200
        mock_settings.chunk_overlap = 50
        mock_get_settings.return_value = mock_settings

        mock_embeddings = MagicMock(spec=HuggingFaceEmbeddings)

        from app.services.data_service import DataIngestionService
        service = DataIngestionService(embeddings=mock_embeddings)

        assert service._persist_path == Path("/tmp/vs/hf")

    @patch("app.services.data_service.get_settings")
    def test_persist_path_gemini(self, mock_get_settings):
        """Debe asignar el path de persistencia correcto para Gemini."""
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        mock_settings = MagicMock()
        mock_settings.persist_path_huggingface = Path("/tmp/vs/hf")
        mock_settings.persist_path_gemini = Path("/tmp/vs/gemini")
        mock_settings.chunk_size = 200
        mock_settings.chunk_overlap = 50
        mock_get_settings.return_value = mock_settings

        mock_embeddings = MagicMock(spec=GoogleGenerativeAIEmbeddings)

        from app.services.data_service import DataIngestionService
        service = DataIngestionService(embeddings=mock_embeddings)

        assert service._persist_path == Path("/tmp/vs/gemini")

    @patch("app.services.data_service.get_settings")
    def test_persist_path_no_soportado_lanza_error(self, mock_get_settings):
        """Debe lanzar excepción para tipo de embeddings no soportado."""
        mock_settings = MagicMock()
        mock_settings.chunk_size = 200
        mock_settings.chunk_overlap = 50
        mock_get_settings.return_value = mock_settings

        mock_embeddings = MagicMock()  # sin spec = no es HuggingFace ni Gemini

        from app.services.data_service import DataIngestionService
        with pytest.raises(Exception):
            DataIngestionService(embeddings=mock_embeddings)

    @patch("app.services.data_service.load_documents")
    @patch("app.services.data_service.get_settings")
    def test_load_and_chunk(self, mock_get_settings, mock_load_docs):
        """Debe retornar documentos cargados y chunks."""
        from langchain_huggingface import HuggingFaceEmbeddings

        mock_settings = MagicMock()
        mock_settings.persist_path_huggingface = Path("/tmp/vs/hf")
        mock_settings.chunk_size = 200
        mock_settings.chunk_overlap = 50
        mock_settings.file_path = Path("/tmp/corpus")
        mock_get_settings.return_value = mock_settings

        mock_load_docs.return_value = [
            Document(page_content="Texto largo " * 50, metadata={"source": "test.pdf"})
        ]

        mock_embeddings = MagicMock(spec=HuggingFaceEmbeddings)

        from app.services.data_service import DataIngestionService
        service = DataIngestionService(embeddings=mock_embeddings)
        all_docs, chunks = service.load_and_chunk()

        assert len(all_docs) == 1
        assert len(chunks) >= 1  # debería generarse al menos un chunk

    @patch("app.services.data_service.get_settings")
    def test_load_vector_store_desde_cache(self, mock_get_settings):
        """Debe retornar el vector store si ya está en memoria."""
        from langchain_huggingface import HuggingFaceEmbeddings

        mock_settings = MagicMock()
        mock_settings.persist_path_huggingface = Path("/tmp/vs/hf")
        mock_settings.chunk_size = 200
        mock_settings.chunk_overlap = 50
        mock_get_settings.return_value = mock_settings

        mock_embeddings = MagicMock(spec=HuggingFaceEmbeddings)

        from app.services.data_service import DataIngestionService
        service = DataIngestionService(embeddings=mock_embeddings)

        mock_vs = MagicMock()
        service._vector_store = mock_vs

        result = service.load_vector_store()
        assert result == mock_vs

    @patch("app.services.data_service.FAISS")
    @patch("app.services.data_service.get_settings")
    def test_load_vector_store_desde_disco(self, mock_get_settings, mock_faiss):
        """Debe cargar el vector store desde disco si existe la carpeta."""
        from langchain_huggingface import HuggingFaceEmbeddings

        mock_persist = MagicMock(spec=Path)
        mock_persist.exists.return_value = True

        mock_settings = MagicMock()
        mock_settings.persist_path_huggingface = mock_persist
        mock_settings.chunk_size = 200
        mock_settings.chunk_overlap = 50
        mock_get_settings.return_value = mock_settings

        mock_vs = MagicMock()
        mock_vs.index = MagicMock()
        mock_vs.index.ntotal = 10
        mock_vs.index.d = 384
        mock_faiss.load_local.return_value = mock_vs

        mock_embeddings = MagicMock(spec=HuggingFaceEmbeddings)

        from app.services.data_service import DataIngestionService
        service = DataIngestionService(embeddings=mock_embeddings)
        result = service.load_vector_store()

        mock_faiss.load_local.assert_called_once()
        assert result == mock_vs

    @patch("app.services.data_service.get_settings")
    def test_print_info_sin_vector_store(self, mock_get_settings):
        """Debe loggear warning si el vector store no fue inicializado."""
        from langchain_huggingface import HuggingFaceEmbeddings

        mock_settings = MagicMock()
        mock_settings.persist_path_huggingface = Path("/tmp/vs/hf")
        mock_settings.chunk_size = 200
        mock_settings.chunk_overlap = 50
        mock_get_settings.return_value = mock_settings

        mock_embeddings = MagicMock(spec=HuggingFaceEmbeddings)

        from app.services.data_service import DataIngestionService
        service = DataIngestionService(embeddings=mock_embeddings)

        # No debería lanzar error
        service.print_vector_store_info()
