from typing import List
from pathlib import Path
import logging
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from app.loaders.base import BaseLoader

logger = logging.getLogger(__name__)

class PDFLoader(BaseLoader):
    def __init__(self, path: Path):
        self.path = path.resolve()

    def load(self) -> List[Document]:
        docs: List[Document] = []

        logger.info(f"Cargando PDF: {self.path}")

        try:
            loader = PyMuPDFLoader(str(self.path))
            loaded_docs = loader.load()

            for doc in loaded_docs:
                content = (doc.page_content or "").strip()
                if content:
                    doc.page_content = content
                    doc.metadata["source_type"] = "pdf"
                    docs.append(doc)

            if not docs:
                logger.warning(f"PDF sin contenido: {self.path}")

        except Exception as e:
            logger.error(f"Error cargando PDF {self.path}: {e}")

        return docs
