from typing import List
from langchain_core.documents import Document

def normalize_documents(docs: List[Document]) -> List[Document]:
    normalized: List[Document] = []

    for doc in docs:
        content = (doc.page_content or "").strip()
        
        if not content:
            continue

        normalized.append(
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

    return normalized
