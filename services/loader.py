from typing import List
from langchain_core.documents import Document
import os
from langchain_community.document_loaders import PyMuPDFLoader
# import bs4
# from langchain_community.document_loaders import WebBaseLoader

def load_pdf_documents(file_path: str) -> List[Document]:
    docs: List[Document] = []
    file_path = os.path.abspath(file_path)

    try:
        loader = PyMuPDFLoader(file_path)
        loaded_docs = loader.load()
        for doc in loaded_docs:
            if doc.page_content and doc.page_content.strip():
                docs.append(doc)

    except Exception as e:
        # archivos corruptos o problemas de lectura
        print(f"No se pudo cargar el archivo PDF: {file_path} | Error: {e}")

    return docs

# def load_web_documents(urls: List[str]) -> List:
#     all_docs: List = []
#     bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
#     web_loader = WebBaseLoader(
#         web_paths=urls,
#         bs_kwargs={"parse_only": bs4_strainer},
#     )
#     all_docs = web_loader.load()
#     return all_docs

def load_documents(path: str) -> List[Document]:
    all_docs: List[Document] = []
    path = os.path.abspath(path)

    if os.path.isdir(path):
        # recorrer directorio y cargar todos los documentos
        for root, _, files in os.walk(path):
            for file in files:
                if file.lower().endswith(".pdf"):
                    full_path = os.path.join(root, file)
                    all_docs.extend(load_pdf_documents(full_path))
                # elif file.lower().startswith("http"):
                #     full_path = os.path.join(root, file)
                #     all_docs.extend(load_web_documents([full_path]))

    elif os.path.isfile(path):
        # cargar un solo archivo
        if path.lower().endswith(".pdf"):
            all_docs.extend(load_pdf_documents(path))
        # elif path.lower().startswith("http"):
        #     return load_web_documents([path])
    else:
        print(f"Archivo no soportado o ruta inválida: {path}")

    return _normalize_documents(all_docs)

# Normalización final
def _normalize_documents(docs: List[Document]) -> List[Document]:
    normalized_docs: List[Document] = []

    for doc in docs:
        title = doc.metadata.get("title") or ""
        source = doc.metadata.get("source")
        page = doc.metadata.get("page")

        normalized_docs.append(
            Document(
                page_content=doc.page_content.strip(),
                metadata={
                    "title": title,
                    "source": source,
                    "page": page
                }
            )
        )

    return normalized_docs

if __name__ == "__main__":
    docs = load_documents("corpus")
    print(f"Documentos cargados: {len(docs)}")

    print("--- VISTA PREVIA DE DOCUMENTOS CARGADOS ---")
    # print(docs[0].metadata['title'])
    # print(docs[0].page_content[:500])
    print(docs[0])