from typing import List
from langchain_core.documents import Document
import os
from langchain_community.document_loaders import PyMuPDFLoader
# import bs4
# from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf_documents(file_path: str) -> List[Document]:
    all_docs: List[Document] = []
    file_path = os.path.abspath(file_path)

    if os.path.isfile(file_path):
        if file_path.lower().endswith(".pdf"):
            try:
                loader = PyMuPDFLoader(file_path)
                for doc in docs:
                    if doc.page_content and doc.page_content.strip():
                        all_docs.append(doc)
            except Exception:
                # archivos corruptos o problemas de lectura
                print(f"No se pudo cargar el archivo PDF: {file_path}")
    elif os.path.isdir(file_path):
        for root, _, files in os.walk(file_path):
            for file in files:
                if file.lower().endswith(".pdf"):
                    full_path = os.path.join(root, file)
                    try:
                        loader = PyMuPDFLoader(full_path)
                        for doc in docs:
                            if doc.page_content and doc.page_content.strip():
                                all_docs.append(doc)
                        docs = loader.load()
                    except Exception:
                        # archivos corruptos o problemas de lectura
                        print(f"No se pudo cargar el archivo PDF: {full_path}")

    return all_docs

# def load_web_documents(urls: List[str]) -> List:
#     all_docs: List = []
#     bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
#     web_loader = WebBaseLoader(
#         web_paths=urls,
#         bs_kwargs={"parse_only": bs4_strainer},
#     )
#     all_docs = web_loader.load()
#     return all_docs

def load_documents(file_path: str) -> List:
    if file_path.lower().endswith(".pdf") or os.path.isdir(file_path):
        return load_pdf_documents(file_path)
    # elif file_path.startswith("http"):
    #     return load_web_documents([file_path])
    else:
        print(f"Tipo de archivo no soportado o ruta inválida: {file_path}")
        return []
