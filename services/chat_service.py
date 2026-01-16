class ChatService:
    def __init__(self, vector_store, retriever, chat_model):
        self.vector_store = vector_store
        self.retriever = retriever
        self.chat_model = chat_model
