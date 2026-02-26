"""Tests para app/loaders/loader.py"""
from unittest.mock import patch, MagicMock
from pathlib import Path
from langchain_core.documents import Document
import pytest


class TestLoadDocuments:
    """Tests para la función load_documents."""

    @patch("app.loaders.loader.normalize_documents", side_effect=lambda x: x)
    @patch("app.loaders.loader.PDFLoader")
    @patch("app.loaders.loader.settings")
    def test_carga_directorio_con_pdfs(self, mock_settings, mock_pdf_class, mock_normalize):
        """Debe cargar PDFs de un directorio recursivamente."""
        mock_settings.urls = []

        # Crear directorio temporal con un PDF
        with patch.object(Path, "resolve", return_value=Path("/tmp/corpus")), \
             patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "is_dir", return_value=True), \
             patch.object(Path, "glob", return_value=[Path("/tmp/corpus/doc.pdf")]):

            mock_loader = MagicMock()
            mock_loader.load.return_value = [
                Document(page_content="PDF content", metadata={"source": "doc.pdf"})
            ]
            mock_pdf_class.return_value = mock_loader

            from app.loaders.loader import load_documents
            docs = load_documents("/tmp/corpus", include_web=False)

        assert len(docs) == 1

    @patch("app.loaders.loader.normalize_documents", side_effect=lambda x: x)
    @patch("app.loaders.loader.PDFLoader")
    @patch("app.loaders.loader.settings")
    def test_carga_archivo_pdf_individual(self, mock_settings, mock_pdf_class, mock_normalize):
        """Debe cargar un archivo PDF individual."""
        mock_settings.urls = []

        path = Path("/tmp/doc.pdf")
        with patch.object(Path, "resolve", return_value=path), \
             patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "is_dir", return_value=False), \
             patch.object(Path, "is_file", return_value=True):

            # suffix es propiedad real de Path, no necesita mock
            mock_loader = MagicMock()
            mock_loader.load.return_value = [
                Document(page_content="Contenido", metadata={})
            ]
            mock_pdf_class.return_value = mock_loader

            from app.loaders.loader import load_documents
            docs = load_documents("/tmp/doc.pdf", include_web=False)

        assert len(docs) == 1

    @patch("app.loaders.loader.normalize_documents", side_effect=lambda x: x)
    @patch("app.loaders.loader.settings")
    def test_ruta_no_soportada(self, mock_settings, mock_normalize):
        """No debe cargar nada si la ruta no es PDF ni directorio."""
        mock_settings.urls = []

        path = Path("/tmp/file.txt")
        with patch.object(Path, "resolve", return_value=path), \
             patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "is_dir", return_value=False), \
             patch.object(Path, "is_file", return_value=True):

            from app.loaders.loader import load_documents
            docs = load_documents("/tmp/file.txt", include_web=False)

        assert len(docs) == 0

    @patch("app.loaders.loader.normalize_documents", side_effect=lambda x: x)
    @patch("app.loaders.loader.WebLoader")
    @patch("app.loaders.loader.settings")
    def test_incluye_documentos_web(self, mock_settings, mock_web_class, mock_normalize):
        """Debe incluir documentos web cuando include_web=True y hay URLs."""
        mock_settings.urls = ["https://example.com"]

        mock_web_loader = MagicMock()
        mock_web_loader.load.return_value = [
            Document(page_content="Web content", metadata={"source": "https://example.com"})
        ]
        mock_web_class.return_value = mock_web_loader

        with patch.object(Path, "resolve", return_value=Path("/tmp/corpus")), \
             patch.object(Path, "exists", return_value=False):

            from app.loaders.loader import load_documents
            docs = load_documents("/tmp/corpus", include_web=True)

        assert len(docs) == 1
        mock_web_class.assert_called_once_with(["https://example.com"])

    @patch("app.loaders.loader.normalize_documents", side_effect=lambda x: x)
    @patch("app.loaders.loader.WebLoader")
    @patch("app.loaders.loader.settings")
    def test_no_incluye_web_si_deshabilitado(self, mock_settings, mock_web_class, mock_normalize):
        """No debe incluir documentos web cuando include_web=False."""
        mock_settings.urls = ["https://example.com"]

        with patch.object(Path, "resolve", return_value=Path("/tmp/corpus")), \
             patch.object(Path, "exists", return_value=False):

            from app.loaders.loader import load_documents
            docs = load_documents("/tmp/corpus", include_web=False)

        mock_web_class.assert_not_called()

    @patch("app.loaders.loader.normalize_documents")
    @patch("app.loaders.loader.settings")
    def test_normaliza_documentos_de_salida(self, mock_settings, mock_normalize):
        """El output debe pasar por normalize_documents."""
        mock_settings.urls = []
        mock_normalize.return_value = [
            Document(page_content="Normalizado", metadata={})
        ]

        with patch.object(Path, "resolve", return_value=Path("/tmp/corpus")), \
             patch.object(Path, "exists", return_value=False):

            from app.loaders.loader import load_documents
            docs = load_documents("/tmp/corpus", include_web=False)

        mock_normalize.assert_called_once()
        assert docs[0].page_content == "Normalizado"
