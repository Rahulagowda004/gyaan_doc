import os
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

def available_pdfs() -> list:
    """
    Returns a list of all available PDF files in the current directory.
    """
    pdf_files = [f for f in os.listdir('R:/gyaan_doc/pdfs') if f.endswith('.pdf')]
    return pdf_files

embeddings = OllamaEmbeddings(model = "all-minilm:latest")

def get_retriever():
    pdfs = available_pdfs()
    for pdf in pdfs:
        pdf_loader = PyPDFLoader(os.path.join("R:/gyaan_doc/pdfs", pdf))

    try:
        pages = pdf_loader.load()
        print(f"PDF has been loaded and has {len(pages)} pages")
    except Exception as e:
        print(f"Error loading PDF: {e}")
        raise
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    pages_split = text_splitter.split_documents(pages) 

    persist_directory = r"embeddings"
    collection_name = "pdf_embeddings"

    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    try:
        vectorstore = Chroma.from_documents(
            documents=pages_split,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        print(f"Created ChromaDB vector store!")
        
    except Exception as e:
        print(f"Error setting up ChromaDB: {str(e)}")
        raise

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )
    
    return retriever