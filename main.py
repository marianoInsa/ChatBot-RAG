from fastapi import FastAPI
import uvicorn
from services.loader import load_pdf_documents
from langchain_ollama import ChatOllama
import faiss
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes

app = FastAPI(
    title="Chatbot RAG Server",
    version="1.0",
    description="Chatbot RAG usando LangChain",
)

# 3. CONFIG DEL RETRIEVER
retriever = vector_store.as_retriever(
    search_type="mmr", # Maximal Marginal Relevance
    search_kwargs={
        "k": 5, # numero de documentos a devolver
        "fetch_k": 20, # numero de documentos a considerar para MMR
        "lambda_mult": 0.5 # 0.5 balancea entre similarity vs diversity
    }
)

# 4. MODELO LLM (Ollama)
llm = ChatOllama(
    model="llama2",
    validate_model_on_init=True,
    temperature=0.2, # subirlo lo hace mas creativo
    # seed=77, # setear una seed hace que las respuestas sean reproducibles
    num_predict=256,

)

# 5. RAG CHAIN
system_template = """Eres un Asistente Virtual experto de la mueblería "Hermanos Jota". 
Tu trabajo es responder en español a preguntas sobre la mueblería de manera profesional y comercial.

Instrucciones:
1. Usa SOLO el contexto proporcionado abajo para responder. No inventes información.
2. Si la respuesta no está en el contexto, di amablemente que no tienes esa información y sugiere contactar a ventas.
3. Sé breve y conciso. Evita introducciones largas como "Basado en el contexto...". Ve al grano.
4. Mantén un tono cordial y servicial.

<context>
{context}
</context>

Pregunta del cliente: {input}
"""
prompt = ChatPromptTemplate.from_template(system_template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    RunnableParallel(
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
    )
    | prompt
    | llm
    | StrOutputParser()
)

print("Cadena RAG lista.")

# 6. SERVIDOR
add_routes(
    app, 
    rag_chain, 
    path="/rag"
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)