"""Tests para app/loaders/normalizer.py"""
from langchain_core.documents import Document
import pytest

from app.loaders.normalizer import normalize_documents


class TestNormalizeDocuments:
    """Tests para la función normalize_documents."""

    def test_filtra_documentos_vacios(self):
        """Debe descartar documentos con contenido vacío."""
        docs = [
            Document(page_content="Contenido válido", metadata={"source": "a"}),
            Document(page_content="", metadata={"source": "b"}),
            Document(page_content="   ", metadata={"source": "c"}),
        ]
        result = normalize_documents(docs)

        assert len(result) == 1
        assert result[0].page_content == "Contenido válido"

    def test_elimina_whitespace(self):
        """Debe limpiar whitespace al inicio y final del contenido."""
        docs = [
            Document(page_content="  texto con espacios  ", metadata={"source": "a"}),
        ]
        result = normalize_documents(docs)

        assert result[0].page_content == "texto con espacios"

    def test_preserva_metadata(self):
        """Debe copiar title, source, page y source_type a la metadata normalizada."""
        docs = [
            Document(
                page_content="Contenido",
                metadata={
                    "title": "Mi Título",
                    "source": "test.pdf",
                    "page": 3,
                    "source_type": "pdf",
                    "extra_field": "ignorado"
                }
            ),
        ]
        result = normalize_documents(docs)

        assert result[0].metadata["title"] == "Mi Título"
        assert result[0].metadata["source"] == "test.pdf"
        assert result[0].metadata["page"] == 3
        assert result[0].metadata["source_type"] == "pdf"
        # extra_field no se incluye en la normalización
        assert "extra_field" not in result[0].metadata

    def test_lista_vacia_retorna_vacia(self):
        """Debe retornar lista vacía si la entrada está vacía."""
        result = normalize_documents([])
        assert result == []

    def test_metadata_parcial(self):
        """Debe manejar metadata incompleta sin errores."""
        docs = [
            Document(page_content="Sin metadata", metadata={}),
        ]
        result = normalize_documents(docs)

        assert len(result) == 1
        assert result[0].metadata["title"] == ""
        assert result[0].metadata["source"] is None
        assert result[0].metadata["page"] is None
        assert result[0].metadata["source_type"] is None
