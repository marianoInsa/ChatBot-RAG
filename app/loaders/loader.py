from pathlib import Path
from typing import List
from langchain_core.documents import Document
import logging

from app.loaders.pdf import PDFLoader
from app.loaders.web import WebLoader
from app.loaders.normalizer import normalize_documents
from app.config.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

def load_documents(path: str, include_web: bool = False) -> List[Document]:
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

if __name__ == "__main__":
    docs = load_documents("corpus", include_web=True)
    print(docs)