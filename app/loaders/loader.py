from pathlib import Path
from typing import List, Union
from langchain_core.documents import Document
import logging

from app.loaders.pdf import PDFLoader
from app.loaders.web import WebLoader
from app.loaders.normalizer import normalize_documents
from app.config.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

def load_documents(path: str, include_web: bool = True) -> List[Document]:
    path = Path(path).resolve()
    all_docs: List[Document] = []

    # CARGA LOCAL
    logger.info(f"Iniciando carga local desde: {path}")
    try:
        if path.exists():
            if path.is_dir():
                for pdf in path.glob("**/*.pdf"):
                    all_docs.extend(PDFLoader(pdf).load())
            elif path.is_file() and path.suffix.lower() == ".pdf":
                all_docs.extend(PDFLoader(path).load())
            else:
                logger.warning(f"Ruta no soportada: {path}")
        logger.info(f"Carga local completada. Documentos cargados: {len(all_docs)}")
    except Exception as e:
        logger.error(f"Error durante la carga local: {e}")

    # CARGA WEB
    if include_web and settings.urls:
        logger.info(f"Iniciando carga Web de {len(settings.urls)} URLs...")
        try:
            web_loader = WebLoader(settings.urls)
            web_docs = web_loader.load()

            if web_docs:
                logger.info(f"Carga web finalizada: {len(web_docs)} documentos obtenidos.")
                all_docs.extend(web_docs)
            else:
                logger.warning("La carga web finalizó sin obtener documentos.")
        except Exception as e:
            logger.error(f"Error durante la carga web: {e}")

    return normalize_documents(all_docs)


def load_documents_from_sources(
    pdf_paths: List[Union[Path, str]] | None = None,
    urls: List[str] | None = None,
) -> List[Document]:
    """
    Carga documentos desde rutas de archivos PDF y/o URLs.
    Acepta fuentes dinámicas (archivos subidos, URLs proporcionadas por el usuario).
    """
    all_docs: List[Document] = []

    if pdf_paths:
        for path in pdf_paths:
            path = Path(path).resolve()
            if path.is_file() and path.suffix.lower() == ".pdf":
                try:
                    all_docs.extend(PDFLoader(path).load())
                except Exception as e:
                    logger.error(f"Error cargando PDF {path}: {e}")

    if urls:
        urls = [u.strip() for u in urls if u and u.strip()]
        if urls:
            try:
                web_loader = WebLoader(urls)
                web_docs = web_loader.load()
                if web_docs:
                    all_docs.extend(web_docs)
            except Exception as e:
                logger.error(f"Error durante la carga web: {e}")

    return normalize_documents(all_docs)


if __name__ == "__main__":
    docs = load_documents("corpus", include_web=True)
    print(docs)