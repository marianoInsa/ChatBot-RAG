# import bs4
from fastapi import FastAPI
import uvicorn
# from langchain_community.llms import Ollama
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader
import faiss
# from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes

app = FastAPI(
    title="Chatbot RAG Server",
    version="1.0",
    description="Chatbot RAG usando LangChain, Ollama y FastAPI",
)

# 1. CARGA DE DATOS
print("Cargando base de conocimiento...")

# bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
# web_loader = WebBaseLoader(
#     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
#     bs_kwargs={"parse_only": bs4_strainer},
# )
# web_docs = web_loader.load()

pdf_loader_1 = PyMuPDFLoader("corpus\catalogo.pdf")
pdf_loader_2 = PyMuPDFLoader("corpus\manual-de-marca.pdf")
pdf_docs_1 = pdf_loader_1.load()
pdf_docs_2 = pdf_loader_2.load()

all_docs = pdf_docs_1 + pdf_docs_2
print(f"Total de páginas cargadas: {len(all_docs)}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200,
    add_start_index=True
)
all_splits = text_splitter.split_documents(all_docs)
print(f"Corpus splitteado en {len(all_splits)} sub-documentos.")

# 2. EMBEDDINGS
print("Creando embeddings...")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") 

# embedding_dim = len(embeddings.embed_documents(all_splits)[0])
# index = faiss.IndexFlatL2(embedding_dim)

# vector_store = FAISS(
#     embedding_function=embeddings,
#     index=index,
#     docstore=InMemoryDocstore(),
#     index_to_docstore_id={},
# )

print("Indexando chunks...")
vector_store = FAISS.from_documents(documents=all_splits, embedding=embeddings)
print("Embeddings creados.")

print("Configurando cadena RAG...")

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
Tu trabajo es responder preguntas sobre la mueblería de manera profesional y comercial.

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