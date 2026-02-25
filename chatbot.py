from langchain_community.document_loaders import TextLoader
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import dotenv_values

env_values = dotenv_values("./app.env")
open_api_key = env_values['OPEN_API_KEY']

def load_text_file(file_path: str):
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    loader = TextLoader(file_path, encoding="utf-8")
    document = loader.load()
    return document


def chunking(documents, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    return chunks


def get_embeddings(model_name: str = "all-MiniLM-L6-v2"):
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    return embedding_model


def create_vector_store(embedding, chunks):
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory="./chroma.db"
    )
    return store


def format_docs(docs):  #transforms retrieved document objects into one clean, structured context string that the LLM can understand.
    return "\n\n".join(doc.page_content for doc in docs)

docs = load_text_file("Mariam.txt") 
chunks = chunking(docs, 500, 80) 
embedding_model = get_embeddings() 
store = create_vector_store(embedding_model, chunks)

def chat_build():

    # Load embeddings
    embedding_model = get_embeddings()

    # Load existing DB (DO NOT rebuild)
    store = Chroma(
        persist_directory="./chroma.db",
        embedding_function=embedding_model
    )

    retriever = store.as_retriever(search_kwargs={"k": 8})

    llm = ChatOpenAI(
        model_name="google/gemma-3n-e2b-it:free",
        temperature=0.5,
        openai_api_key=open_api_key,
        openai_api_base="https://openrouter.ai/api/v1"
    )

    prompt = ChatPromptTemplate.from_template("""
    You are a question-answering assistant.

    STRICT RULES:
    - Answer ONLY using the provided context.
    - If the answer is NOT explicitly written in the context, say exactly:
      "I do not know."
    - Do NOT use outside knowledge.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
chain = chat_build()
#A try
print(chain.invoke("What had Mariam studied?"))
    