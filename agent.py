import os
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage, BaseMessage, ToolMessage
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, Sequence, TypedDict, Literal
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.tools import tool
import logging # Added for logging

load_dotenv()
logger = logging.getLogger(__name__) # Added logger

# LLM and Embeddings initialized early as they are fundamental
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")
embeddings = OllamaEmbeddings(model = "all-minilm:latest")

# Global retriever dictionary to store user-specific retrievers
global_retrievers = {}

def get_user_retriever(username: str):
    """Get or create a retriever for a specific user."""
    global global_retrievers
    if username not in global_retrievers:
        global_retrievers[username] = initialize_retriever(username)
    return global_retrievers[username]

def reinitialize_retriever(username: str):
    """Reinitialize the retriever for a specific user."""
    global global_retrievers
    global_retrievers[username] = initialize_retriever(username)
    return global_retrievers[username]

def initialize_retriever(username: str):
    """Loads PDFs, processes them, creates ChromaDB vector store for a specific user."""
    logger.info(f"Initializing retriever for user: {username}")
    pdf_files_dir = os.path.join('R:/gyaan_doc/pdfs', username)
    if not os.path.exists(pdf_files_dir):
        logger.warning(f"PDFs directory not found for user {username}: {pdf_files_dir}")
        return None

    pdf_files = [f for f in os.listdir(pdf_files_dir) if f.endswith('.pdf')]
    if not pdf_files:
        logger.warning(f"No PDF files found for user {username}")
        return None
        
    all_pages = []

    for pdf_file in pdf_files:
        logger.info(f"Processing PDF for user {username}: {pdf_file}")
        pdf_loader = PyPDFLoader(os.path.join(pdf_files_dir, pdf_file))
        try:
            pages = pdf_loader.load()
            all_pages.extend(pages)
        except Exception as e:
            logger.error(f"Error loading PDF {pdf_file} for user {username}: {e}", exc_info=True)
            continue

    logger.info(f"Total pages from all PDFs for user {username}: {len(all_pages)}")

    if not all_pages:
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=16000,
        chunk_overlap=750
    )
    pages_split = text_splitter.split_documents(all_pages)
    
    if not pages_split:
        return None

    persist_directory = os.path.join("Agents", username)
    collection_name = f"pdfs_embeddings_{username}"
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    try:
        vectorstore = Chroma.from_documents(
            documents=pages_split,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        logger.info(f"ChromaDB vector store created/loaded for user {username}.")
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
    except Exception as e:
        logger.error(f"Error setting up ChromaDB for user {username}: {str(e)}", exc_info=True)
        return None

@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns the information from all PDF documents in the database.
    You must use this tool before providing any response.
    """
    # Get username from the tool's context (passed through config)
    username = tools_context.get('username', 'default')
    retriever = get_user_retriever(username)
    
    if retriever is None:
        logger.error(f"Retriever is not initialized for user: {username}")
        return "Error: Document retriever is not available."

    logger.info(f"Retriever tool called by user {username} with query: {query[:50]}...")
    docs = retriever.invoke(query)
    
    if not docs:
        return "I found no relevant information in the available PDF documents."
    
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")
    
    return "\n\n".join(results)

# Initialize tools context
tools_context = {'username': 'default'}

def set_tools_context(username: str):
    """Set the context for tools to use."""
    global tools_context
    tools_context['username'] = username

# Remove duplicate retriever_tool and global_retriever initialization
tools = [retriever_tool]
llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

system_prompt = """
You are an intelligent AI assistant that MUST ALWAYS use the retriever tool before providing any response. Follow these rules strictly:

1. ALWAYS start by using the retriever tool to search for relevant information
2. Your first action for EVERY query must be to use the retriever tool
3. If the retriever returns no results, try a rephrased search with different keywords
4. Only use information found in the documents through the retriever tool
5. DO NOT use any prior knowledge or make assumptions
6. If after multiple attempts you cannot find relevant information, clearly state that the information is not found in the documents

Your responses should follow this structure:
1. Search the documents using the retriever tool
2. If needed, perform additional searches with different keywords
3. Synthesize the retrieved information into a response
4. If no relevant information is found, clearly state this fact

Remember: You must ALWAYS use the retriever tool at least once for EVERY query, regardless of the question type.
"""

tools_dict = {our_tool.name: our_tool for our_tool in tools}

def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    messages = list(state['messages'])
    # Prepend system prompt only if it's not already effectively there
    # This basic check might need refinement for more complex histories
    if not any(isinstance(m, SystemMessage) and m.content == system_prompt for m in messages):
         messages = [SystemMessage(content=system_prompt)] + messages
    
    # Use the LLM that has tools bound to it
    message = llm_with_tools.invoke(messages)
    return {'messages': [message]}

def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""
    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        logger.info(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        
        if not t['name'] in tools_dict:
            logger.warning(f"Tool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        else:
            tool_input = t['args'].get('query', '') # Ensure tool is called with its expected arg name
            if t['name'] == retriever_tool.name and not isinstance(tool_input, str): # Basic check for retriever
                tool_input = str(tool_input) # Ensure query is a string for retriever_tool
            result = tools_dict[t['name']].invoke(tool_input) 
            logger.info(f"Result length from tool {t['name']}: {len(str(result))}")
            
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    logger.info("Tools Execution Complete. Back to the model!")
    return {'messages': results}

memory = MemorySaver()
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent", False: END}
)
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

app = graph.compile(checkpointer=memory)

# Export the app at the end of the file
__all__ = ['app', 'set_tools_context', 'reinitialize_retriever']

if __name__ == "__main__":
    # This block will only execute if agent.py is run directly, not when imported.
    config = {"configurable": {"thread_id": "test-cli-1"}} # Use a distinct thread_id for CLI testing
    def main_test_loop():
        logger.info("Starting agent test loop for direct execution...")
        while True:
            user_input = input("Enter your query (or type 'exit' to quit): ")
            if user_input.lower() == 'exit':
                break

            logger.info(f"User query: {user_input}")
            events = app.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config,
                stream_mode="values",
            )
            for event in events:
                if event and "messages" in event and event["messages"]:
                    # Log the full message for debugging if needed
                    # logger.debug(f"Agent event message: {event['messages'][-1]}")
                    event["messages"][-1].pretty_print() # For CLI, pretty_print is fine
    main_test_loop()