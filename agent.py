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

