from langchain_ollama import ChatOllama

llama2 = ChatOllama(
    model="llama2",
    validate_model_on_init=True,
    temperature=0.2, # subirlo lo hace mas creativo
    # seed=77, # setear una seed hace que las respuestas sean reproducibles
    num_predict=256,

)