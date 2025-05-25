import os
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage, BaseMessage
from utils import available_pdfs
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, Sequence, TypedDict, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from utils import get_retriever
from langchain import hub

load_dotenv()

####state of the graph
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]
    pdf_query: list[str]
    
####llm
llm = ChatGroq(model = "llama-3.3-70b-versatile",max_tokens=16000)

####class for structured output for orchestrator
class Parser(BaseModel):
    """LLM will be allowed to output only 'summarization' or 'rag' based on the user query."""
    query: Literal["summarization", "rag"]
    pdfs: list[str] = Field(...,description="pdfs mentioned in the user query, if any")

####agent orchestrator function
def orchestrator(state: State) -> State:
    """Router function to determine the next step in the state graph.
    This function checks the current state and decides which tool or action to take next.

    Args:
        state (State): The current state of the graph.

    Returns:
        State: The updated state after processing the current step.
    """
    
    prompt = PromptTemplate(
    template="""
        You are a decision-making agent. Based on the user's query, you must select:

        1. <summarization> — Use this **only if** the user asks for a summary of the **entire PDF**.
        2. <rag> — Use this for **any other type of query**, such as questions about specific sections, topics, paragraphs, or details from the PDF.
        3. <pdfs> — Choose the **relevant PDF filenames** from the provided list that best match the user query. If none are relevant, return an **empty list**.

        Your response must include:
        - One tag: `<summarization>` or `<rag>`
        - A list of relevant PDFs in the `<pdfs>` tag

        Format:
        <decision>
        <pdfs>[list of relevant PDFs]</pdfs>

        User Query: {input}

        Available PDFs: {pdfs}
        """,
            input_variables=["input", "pdfs"],
        )
    
    structured_llm = llm.with_structured_output(Parser)
    
    chain = prompt | structured_llm
    
    pdfs = available_pdfs()
    response = chain.invoke({
        'input': state["messages"], 
        'pdfs': pdfs
    })
      # Access Parser object fields with dot notation
    query_type = response.query  # This will be either "summarization" or "rag"
    selected_pdfs = response.pdfs  # This will be a list of PDFs
    
    print(f"Query type: {query_type}")
    print(f"Selected PDFs: {selected_pdfs}")
    
    # Add the routing decision as an AI message and include selected PDFs in the state
    return {
        "messages": state["messages"] + [AIMessage(content=query_type)],
        "pdf_query": selected_pdfs
    }

####summarization agent
def summarization(state: State) -> State:
    """Summarize the content of the selected PDFs."""
    try:
        # Get the list of PDFs from the state
        pdfs = state.get("pdf_query", [])
        
        print(f"Summarization agent received PDFs: {pdfs}")
        
        if not pdfs:
            pdfs = available_pdfs()
            return pdfs
        
        all_docs = []
        for pdf in pdfs:
            pdf_path = os.path.join("R:/gyaan_doc/pdfs", pdf)
            print(f"Loading PDF from: {pdf_path}")
            
            loader = PyPDFLoader(Path(pdf_path))
            docs = loader.load_and_split()
            print(f"Loaded {len(docs)} document chunks from {pdf}")
            all_docs.extend(docs)
        
        # If no documents were loaded, return early
        if not all_docs:
            return {"messages": [AIMessage(content="Could not load any documents for summarization.")]}
        
        print(f"Total document chunks: {len(all_docs)}")
        
        split_docs = RecursiveCharacterTextSplitter(chunk_size=15000, chunk_overlap=500).split_documents(all_docs)
        print(f"After splitting: {len(split_docs)} chunks")
        
        chain = load_summarize_chain(
            llm=llm,
            chain_type="refine",
            verbose=True,  # Set to True for debugging
        )
        
        print("Starting summarization chain...")
        response = chain.invoke({"input_documents": split_docs}, return_only_outputs=True)
        
        print("Summarization complete. Returning the output.")
        
        if isinstance(response, dict) and 'output_text' in response:
            output = response['output_text']
        else:
            output = str(response)
        
        return {"messages": [AIMessage(content=output)]}
    except Exception as e:
        print(f"Error in summarization agent: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"messages": [AIMessage(content=f"An error occurred during summarization: {str(e)}")]}


####rag agent
retriever_instance = get_retriever()

def rag(state: State) -> State:
    """Execute tool calls from the LLM's response."""
    try:
        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        combine_docs_chain = create_stuff_documents_chain(
            llm, retrieval_qa_chat_prompt
        )
        retrieval_chain = create_retrieval_chain(retriever_instance, combine_docs_chain)
        
        messages = state["messages"]
        # We need to find the actual user query, not the routing decision
        # Filter out messages that contain our routing keywords
        user_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
        if not user_messages:
            return {'messages': [AIMessage(content="No user query found.")]}
            
        # Use the last user message
        last_user_message = user_messages[-1]
        
        print(f"Processing RAG query: {last_user_message.content}")
        
        results = retrieval_chain.invoke({
            "input": last_user_message.content,
        })
        
        # Print the structure of results to debug
        print(f"Results structure: {type(results)}")
        if isinstance(results, dict):
            print(f"Results keys: {list(results.keys())}")
        
        # Try different ways to extract the answer
        if isinstance(results, str):
            answer = results
        elif isinstance(results, dict):
            if "answer" in results:
                answer = results["answer"]
            elif "result" in results:
                answer = results["result"]
            elif "output" in results:
                answer = results["output"]
            elif "response" in results:
                answer = results["response"]
            elif "content" in results:
                answer = results["content"]
            elif "output_text" in results:
                answer = results["output_text"]
            else:
                # As a fallback, convert the entire results dict to string
                answer = str(results)
        else:
            answer = str(results)
        
        print(f"RAG answer type: {type(answer)}")
        
        return {'messages': [AIMessage(content=answer)]}
    except Exception as e:
        print(f"Error in RAG agent: {str(e)}")
        return {'messages': [AIMessage(content=f"An error occurred: {str(e)}")]}


####function helps to decide whether to go with rag or summarization
def to_continue(state: State) -> State:
    """This function takes the output from router and decided what do no next."""
    messages = state["messages"]
    last_message = messages[-1]
    
    if isinstance(last_message, AIMessage):
        if last_message.content == "summarization":
            return "summarization"
        elif last_message.content == "rag":
            return "rag"
        else:
            raise ValueError(f"Unexpected content in last message: {last_message.content}")


####state graph
memory = MemorySaver()

graph = StateGraph(State)

graph.add_node("router",orchestrator)
graph.add_node("rag_agent", rag)
graph.add_node("summarization_agent", summarization)

graph.add_edge(START, "router")
graph.add_conditional_edges(
    "router",
    to_continue,
    {
        "summarization": "summarization_agent",
        "rag": "rag_agent",
    }
)
graph.add_edge("rag_agent", END)
graph.add_edge("summarization_agent", END)

app = graph.compile(checkpointer=memory)

####pinging to the graph
config = {"configurable": {"thread_id": "1"}}

# Example usages
# First try summarization
user_input = "what is the background of rahul a gowda"
print("\n===== Testing summarization =====")
print(f"User query: {user_input}")
events = app.stream(
    {"messages": [HumanMessage(content=user_input)]},
    config,
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()

# # Then try a RAG query
# print("\n===== Testing RAG =====")
# user_input = "what is self attention in the paper?"
# print(f"User query: {user_input}")
# events = app.stream(
#     {"messages": [HumanMessage(content=user_input)]},
#     config,
#     stream_mode="values",
# )
# for event in events:
#     event["messages"][-1].pretty_print()