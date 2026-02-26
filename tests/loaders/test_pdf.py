"""Tests para app/loaders/pdf.py"""
from unittest.mock import patch, MagicMock
from pathlib import Path
from langchain_core.documents import Document
import pytest


class TestPDFLoader:
    """Tests para la clase PDFLoader."""

    @patch("app.loaders.pdf.PyMuPDFLoader")
    def test_carga_pdf_valido(self, mock_pymupdf_class):
        """Debe cargar y retornar documentos de un PDF con contenido."""
        mock_loader = MagicMock()
        mock_loader.load.return_value = [
            Document(page_content="Contenido de la página 1", metadata={"page": 0}),
            Document(page_content="Contenido de la página 2", metadata={"page": 1}),
        ]
        mock_pymupdf_class.return_value = mock_loader

        from app.loaders.pdf import PDFLoader
        loader = PDFLoader(Path("/tmp/test.pdf"))
        docs = loader.load()

        assert len(docs) == 2
        assert docs[0].page_content == "Contenido de la página 1"
        assert docs[0].metadata["source_type"] == "pdf"
        assert docs[1].metadata["source_type"] == "pdf"

    @patch("app.loaders.pdf.PyMuPDFLoader")
    def test_filtra_paginas_vacias(self, mock_pymupdf_class):
        """Debe filtrar documentos con contenido vacío o solo whitespace."""
        mock_loader = MagicMock()
        mock_loader.load.return_value = [
            Document(page_content="Contenido válido", metadata={"page": 0}),
            Document(page_content="   ", metadata={"page": 1}),
            Document(page_content="", metadata={"page": 2}),
        ]
        mock_pymupdf_class.return_value = mock_loader

        from app.loaders.pdf import PDFLoader
        loader = PDFLoader(Path("/tmp/test.pdf"))
        docs = loader.load()

        assert len(docs) == 1
        assert docs[0].page_content == "Contenido válido"

    @patch("app.loaders.pdf.PyMuPDFLoader")
    def test_retorna_lista_vacia_en_error(self, mock_pymupdf_class):
        """Debe retornar lista vacía si ocurre una excepción."""
        mock_pymupdf_class.side_effect = Exception("Error de lectura")

        from app.loaders.pdf import PDFLoader
        loader = PDFLoader(Path("/tmp/test.pdf"))
        docs = loader.load()

        assert docs == []
