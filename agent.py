from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage,ToolMessage, BaseMessage
from utils import available_pdfs
from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import ToolNode
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv

load_dotenv()

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]
    
@tool
def select_pdf(available_pdfs: list,pdfs_in_query: list) -> list:
    """Select PDFs from a list based on user query.
    This tool helps find and select PDF documents that match the user's query 
    based on title, content description, or other metadata.

    Args:
        available_pdfs (list): List of available PDF documents
        pdfs_in_query (list): List of PDF names mentioned in the user query

    Returns:
        list: List of selected PDF documents matching the query, with their metadata.
            Returns empty list if no matches found.
    """
    available_pdfs = available_pdfs()
    if not pdfs_in_query:
        return available_pdfs
    
    selected_pdfs = []

    for pdf in available_pdfs:
        if pdf['title'] in pdfs_in_query:
            selected_pdfs.append(pdf)

    return selected_pdfs