from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader, SeleniumURLLoader
from pathlib import Path
import logging
from services.config import get_settings

logger = logging.getLogger(__name__)

class DocumentLoader:
    def __init__(self):
        self.settings = get_settings()

    def load_pdf_documents(self, file_path: str) -> List[Document]:
        docs: List[Document] = []
        file_path = Path(file_path).resolve()

        logger.info(f"Cargando archivos PDF: {file_path}")

        try:
            loader = PyMuPDFLoader(str(file_path))
            loaded_docs = loader.load()

            for doc in loaded_docs:
                content = (doc.page_content or "").strip()
                if content:
                    doc.page_content = content
                    doc.metadata["source_type"] = "pdf"
                    docs.append(doc)

            if not docs:
                logger.warning(f"El archivo PDF no devolvió contenido: {file_path}")

            logger.info(f"Cargadas {len(docs)} páginas del PDF: {file_path}")

        except Exception as e:
            # archivos corruptos o problemas de lectura
            logger.error(f"No se pudo cargar el archivo PDF: {file_path} | Error: {e}")
            return []

        return docs

    def load_web_documents(self) -> List[Document]:
        docs: List[Document] = []

        urls = self.settings.urls
        if not urls or not isinstance(urls, list):
            logger.warning("No hay URLs configuradas")
            return []
        
        logger.info(f"Cargando documentos web desde: {urls}")

        try:
            web_loader = SeleniumURLLoader(urls)
            raw_docs = web_loader.load()

            if not raw_docs:
                logger.warning(f"El documento web no devolvió contenido")

            for doc in raw_docs:
                content = (doc.page_content or "").strip()

                if not content:
                    continue
                
                doc.page_content = content
                doc.metadata["source_type"] = "web"
                docs.append(doc)

        except Exception as e:
            logger.error(f"Error cargando documentos web. Error: {e}")
            return []

        logger.info(f"Total de documentos web cargados: {len(docs)}")
        return docs

    def load_documents(self, path: str) -> List[Document]:
        all_docs: List[Document] = []
        path = Path(path).resolve()

        if path.is_dir():
            pdf_files = list(path.glob("**/*.pdf"))

            for pdf_file in pdf_files:
                all_docs.extend(self.load_pdf_documents(pdf_file))

        elif path.is_file():
            # cargar un solo archivo
            if path.suffix.lower() == ".pdf":
                all_docs.extend(self.load_pdf_documents(path))
            else:
                logger.warning(f"Archivo no soportado: {path}")

        else:
            logger.warning(f"Archivo no soportado o ruta inválida: {path}")

        all_docs.extend(self.load_web_documents())

        return self._normalize_documents(all_docs)

    # Normalización final
    def _normalize_documents(self, docs: List[Document]) -> List[Document]:
        normalized_docs: List[Document] = []

        for doc in docs:
            content = (doc.page_content or "").strip()

            if not content:
                continue
            
            normalized_docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "title": doc.metadata.get("title", ""),
                        "source": doc.metadata.get("source"),
                        "page": doc.metadata.get("page"),
                        "source_type": doc.metadata.get("source_type")
                    }
                )
            )

        return normalized_docs

if __name__ == "__main__":
    loader = DocumentLoader()
    docs = loader.load_documents("corpus")
    print(f"Documentos cargados desde 'corpus': {docs}")