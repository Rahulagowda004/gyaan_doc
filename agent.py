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

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")

# Our Embedding Model - has to also be compatible with the LLM
embeddings = OllamaEmbeddings(model = "all-minilm:latest")

pdf_files = [f for f in os.listdir('R:/gyaan_doc/pdfs') if f.endswith('.pdf')]

# Initialize an empty list to store all pages from all PDFs
all_pages = []

# Process each PDF file and accumulate all pages
for pdf_file in pdf_files:
    print(f"Processing PDF: {pdf_file}")
    pdf_loader = PyPDFLoader(os.path.join("R:/gyaan_doc/pdfs", pdf_file))  # This loads the PDF with full path
    
    try:
        pages = pdf_loader.load()
        print(f"PDF {pdf_file} has been loaded and has {len(pages)} pages")
        all_pages.extend(pages)  # Add pages from this PDF to our collection
    except Exception as e:
        print(f"Error loading PDF {pdf_file}: {e}")
        # Continue processing other PDFs instead of raising exception
        continue

print(f"Total pages from all PDFs: {len(all_pages)}")

# Chunking Process
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=16000,
    chunk_overlap=750
)

# Split all documents from all PDFs
pages_split = text_splitter.split_documents(all_pages)

persist_directory = r"Agents"
collection_name = "pdfs_embeddings"

# If our collection does not exist in the directory, we create using the os command
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)


try:
    # Here, we actually create the chroma database using our embeddigns model
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


# Now we create our retriever 
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5} # K is the amount of chunks to return
)

@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns the information from all PDF documents in the database.
    """

    docs = retriever.invoke(query)
    
    if not docs:
        return "I found no relevant information in the available PDF documents."
    
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")
    
    return "\n\n".join(results)


tools = [retriever_tool]

llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0


system_prompt = """
You are an intelligent AI assistant that provides answers based solely on the content retrieved using the retriever tool. If the user asks anything related to a PDF, always use the retriever tool to fetch and reference information from the PDF before answering. Do not rely on prior knowledge or assumptions—respond strictly based on retrieved content.
"""

tools_dict = {our_tool.name: our_tool for our_tool in tools} # Creating a dictionary of our tools

# LLM Agent
def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    messages = list(state['messages'])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)
    return {'messages': [message]}


# Retriever Agent
def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        
        if not t['name'] in tools_dict: # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")
            

        # Appends the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Tools Execution Complete. Back to the model!")
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

# config = {"configurable": {"thread_id": "1"}}

# def running_agent():
#     while True:
#         user_input = input("Enter your query (or type 'exit' to quit): ")
#         if user_input.lower() == 'exit':
#             break
# 
#         print("\n===== Testing summarization =====")
#         print(f"User query: {user_input}")
#         events = app.stream(
#             {"messages": [HumanMessage(content=user_input)]},
#             config,
#             stream_mode="values",
#         )
#         for event in events:
#             event["messages"][-1].pretty_print()

# running_agent() # Commented out to prevent execution on import

if __name__ == "__main__":
    # This block will only execute if agent.py is run directly, not when imported.
    config = {"configurable": {"thread_id": "1"}}
    def main_test_loop():
        while True:
            user_input = input("Enter your query (or type 'exit' to quit): ")
            if user_input.lower() == 'exit':
                break

            print("\n===== Agent Test Output =====")
            print(f"User query: {user_input}")
            events = app.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config,
                stream_mode="values",
            )
            for event in events:
                if event and "messages" in event and event["messages"]:
                    event["messages"][-1].pretty_print()
    main_test_loop()