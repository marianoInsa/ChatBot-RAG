class ChatService:
    def __init__(self, vector_store, retriever, chat_model):
        self.vector_store = vector_store
        self.retriever = retriever
        self.chat_model = chat_model

    def data_ingestion(self, documents):
        """Pipeline de ingesta de datos: carga, partición y creación de embeddings."""

        self.vector_store.add_documents(embeddings)
