from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage,ToolMessage, BaseMessage
from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import ToolNode
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv

load_dotenv()

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]
    
@tool
def select_pdf(query: str, available_pdfs: list) -> list:
    """Select PDFs from a list based on user query.
    This tool helps find and select PDF documents that match the user's query 
    based on title, content description, or other metadata.

    Args:
        query (str): User's query or search criteria for finding relevant PDFs.
            Can be keywords, title fragments, or descriptive terms.
        available_pdfs (list): List of available PDF documents, where each item is a dict
            with at least 'title' and optionally 'description', 'author', 'date', etc.

    Returns:
        list: List of selected PDF documents matching the query, with their metadata.
            Returns empty list if no matches found.
    """
    if not available_pdfs:
        return []
    
    # Convert query to lowercase for case-insensitive matching
    query_lower = query.lower()
    selected_pdfs = []
    
    # Search through available PDFs
    for pdf in available_pdfs:
        # Extract searchable fields with fallbacks to empty string if not present
        title = pdf.get('title', '').lower()
        description = pdf.get('description', '').lower()
        author = pdf.get('author', '').lower()
        content = pdf.get('content_preview', '').lower()
        
        # Check if query matches any of the PDF metadata
        if (query_lower in title or 
            query_lower in description or 
            query_lower in author or 
            query_lower in content):
            selected_pdfs.append(pdf)
    
    return selected_pdfs

@tool
def process_pdf(pdf_path: str, chunk_size: int = 1000, overlap: int = 200) -> dict:
    """Process a PDF document by extracting text and splitting it into chunks.
    
    Args:
        pdf_path (str): Path to the PDF file to process
        chunk_size (int, optional): Size of text chunks for processing. Defaults to 1000.
        overlap (int, optional): Overlap between chunks. Defaults to 200.
    
    Returns:
        dict: A dictionary containing:
            - 'text': The full extracted text
            - 'chunks': List of text chunks
            - 'metadata': Document metadata
    """
    try:
        # This is a placeholder for actual PDF processing logic
        # In a real implementation, you would use PyPDF2, pdfplumber, or similar libraries
        import os
        
        if not os.path.exists(pdf_path):
            return {"error": f"File not found: {pdf_path}"}
            
        # Placeholder for PDF text extraction
        # In a real implementation:
        # from pypdf import PdfReader
        # reader = PdfReader(pdf_path)
        # text = ""
        # for page in reader.pages:
        #     text += page.extract_text()
        
        # For demonstration, we'll create mock text
        text = f"This is extracted text from {os.path.basename(pdf_path)}. " * 50
        
        # Split text into chunks
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if chunk:
                chunks.append(chunk)
        
        # Extract basic metadata
        filename = os.path.basename(pdf_path)
        file_size = os.path.getsize(pdf_path) if os.path.exists(pdf_path) else 0
        
        metadata = {
            "filename": filename,
            "file_size": file_size,
            "num_chunks": len(chunks),
            "processing_params": {
                "chunk_size": chunk_size,
                "overlap": overlap
            }
        }
        
        return {
            "text": text,
            "chunks": chunks,
            "metadata": metadata
        }
        
    except Exception as e:
        return {"error": str(e)}

# List of available tools
pdf_tools = [select_pdf, process_pdf]

# Define the agents
def create_decision_maker_agent():
    """Creates the controller agent that routes requests to other agents"""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    
    system_message = """You are a Decision Maker Agent responsible for understanding the user's request 
    and determining which specialized agent should handle it. You have access to these agents:
    1. Document Processor Agent: For extracting and preprocessing PDF documents
    2. RAG Agent: For answering specific questions using retrieval-augmented generation on PDF content
    3. Summarization Agent: For creating summaries of PDF documents at various levels of detail
    
    Based on the user's request, determine which agent should handle the task and explain why.
    """
    
    return llm.bind(system=system_message)

def create_document_processor_agent():
    """Creates the agent that handles document processing tasks"""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    
    system_message = """You are a Document Processor Agent specialized in handling PDF documents.
    Your responsibilities include:
    1. Extracting text from PDFs
    2. Breaking documents into appropriate chunks for processing
    3. Handling different PDF formats and structures
    4. Preparing documents for RAG or summarization tasks
    
    Use the tools available to you to process PDF documents effectively.
    """
    
    return llm.bind(system=system_message)

def create_rag_agent():
    """Creates the agent that handles retrieval-augmented generation tasks"""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    
    system_message = """You are a RAG (Retrieval-Augmented Generation) Agent specialized in answering 
    questions about PDF content using information retrieval techniques and language generation.
    Your tasks include:
    1. Understanding the user's query about document content
    2. Retrieving relevant information from processed documents
    3. Generating accurate answers based on the retrieved content
    4. Citing sources for your answers
    
    Always base your answers on the content of the provided documents.
    """
    
    return llm.bind(system=system_message)

def create_summarization_agent():
    """Creates the agent that handles document summarization tasks"""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    
    system_message = """You are a Summarization Agent specialized in creating concise and informative 
    summaries of PDF documents. Your capabilities include:
    1. Creating executive summaries highlighting key points
    2. Generating detailed chapter-by-chapter summaries
    3. Producing bullet-point lists of main ideas
    4. Adapting summary length and detail based on user requirements
    
    Focus on extracting the most important information while maintaining accuracy.
    """
    
    return llm.bind(system=system_message)

# Enhanced State with tracking capabilities
class EnhancedState(State):
    """Enhanced state that tracks processing history and document states"""
    processed_documents: dict = {}
    selected_documents: list = []
    current_agent: str = "decision_maker"
    processing_history: list = []

# Function to determine which agent should handle the request
def route_to_agent(state: EnhancedState):
    """Determines which agent should handle the request based on the current messages"""
    # Get the last message from the user
    messages = state["messages"]
    last_message = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    
    if not last_message:
        return {"current_agent": "decision_maker"}
    
    # Decision logic based on message content
    content = last_message.content.lower()
    
    if any(kw in content for kw in ["process", "extract", "convert", "open pdf"]):
        return {"current_agent": "document_processor"}
    elif any(kw in content for kw in ["question", "answer", "query", "find", "search"]):
        return {"current_agent": "rag"}
    elif any(kw in content for kw in ["summarize", "summary", "outline", "overview"]):
        return {"current_agent": "summarization"}
    else:
        return {"current_agent": "decision_maker"}

# Create the graph
def create_pdf_processing_graph():
    """Creates the multi-agent system graph for PDF processing"""
    # Initialize agents
    decision_maker = create_decision_maker_agent()
    document_processor = create_document_processor_agent() 
    rag_agent = create_rag_agent()
    summarization_agent = create_summarization_agent()
    
    # Create tool node
    tools_node = ToolNode(pdf_tools)
    
    # Create the graph
    workflow = StateGraph(EnhancedState)
    
    # Add nodes
    workflow.add_node("decision_maker", decision_maker)
    workflow.add_node("document_processor", document_processor)
    workflow.add_node("rag", rag_agent)
    workflow.add_node("summarization", summarization_agent)
    workflow.add_node("tools", tools_node)
    
    # Add edges
    workflow.add_edge("decision_maker", "document_processor")
    workflow.add_edge("decision_maker", "rag")
    workflow.add_edge("decision_maker", "summarization")
    
    workflow.add_edge("document_processor", "tools")
    workflow.add_edge("rag", "tools")
    workflow.add_edge("summarization", "tools")
    workflow.add_edge("tools", "decision_maker")
    
    # Set the entry point
    workflow.set_entry_point("decision_maker")
    
    # Add conditional edges based on routing logic
    workflow.add_conditional_edges(
        "decision_maker",
        route_to_agent,
        {
            "document_processor": "document_processor",
            "rag": "rag",
            "summarization": "summarization",
            "decision_maker": "decision_maker"
        }
    )
    
    # Compile the graph
    return workflow.compile()