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

def create_vector_store():
    pdfs = available_pdfs()
    all_pages = []  # Initialize a list to hold pages from all PDFs
    if not pdfs:
        print("No PDFs found in the 'pdfs' directory.")
        return

    for pdf_filename in pdfs:
        # Corrected path joining
        pdf_path = os.path.join("pdfs", pdf_filename) 
        pdf_loader = PyPDFLoader(pdf_path) # Use the correct path

        try:
            pages = pdf_loader.load()
            print(f"PDF '{pdf_filename}' has been loaded and has {len(pages)} pages")
            all_pages.extend(pages)  # Add loaded pages to the list
        except Exception as e:
            print(f"Error loading PDF '{pdf_filename}': {e}")
            # Optionally, decide whether to continue or raise the exception
            # For now, it will skip the problematic PDF and continue

    if not all_pages:  # Check if any pages were successfully loaded
        print("No documents were loaded from any PDF. Vector store not created.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    pages_split = text_splitter.split_documents(all_pages) # Split documents from all PDFs

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

def get_retriever():
    
    # Check if there are any PDFs to process
    if available_pdfs():
        # Potentially clear old embeddings first if you want a full rebuild
        import shutil
        if os.path.exists("embeddings"):
            shutil.rmtree("embeddings") 
        os.makedirs("embeddings", exist_ok=True) # Ensure directory exists
        
        print("Recreating vector store as PDFs are available.")
        create_vector_store()
    elif not os.listdir("embeddings"):
        print("No PDFs found and no existing embeddings. Cannot create retriever.")
        return None # Or raise an error
        
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=r"embeddings",
        collection_name="pdf_embeddings"
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )
    
    return retriever