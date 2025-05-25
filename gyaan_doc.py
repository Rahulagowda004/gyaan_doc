import streamlit as st
import os
from pathlib import Path
import asyncio
import json
import aiofiles
from datetime import datetime
import time
import logging
# from document_processor import DocumentProcessor # Removed
# from custom_embeddings import CustomEmbedding # Potentially unused if LightRAG is removed
# from custom_llm import get_full_response # Potentially unused if LightRAG is removed
# from lightrag import LightRAG, QueryParam # Potentially unused if LightRAG is removed
# from context_manager import ContextManager # Potentially unused if LightRAG is removed
from PIL import Image
import os
import hashlib
import altair as alt
import pandas as pd

# Authentication functions
def get_url_params():
    """Extract URL query parameters from st.query_params"""
    # Get query parameters using the newer API
    query_params = st.query_params
    
    # With st.query_params, values are direct values, not lists
    user = query_params.get("user", "")
    token = query_params.get("token", "")
    timestamp = query_params.get("ts", "")
    
    # Log the extracted parameters for debugging
    print(f"Extracted URL params - user: {user}, token: {token}, timestamp: {timestamp}")
    
    return user, token, timestamp

def validate_token(username, token, timestamp):
    """Validate the token matches the username"""
    if not username or not token:
        print(f"Missing authentication parameters - user: {username}, token: {token}, timestamp: {timestamp}")
        return False
    try:
        # Remove timestamp/expiration check for static token
        secret_key = "GYAAN_SECRET_KEY_2025"
        token_string = f"{username}:{secret_key}"
        expected_token = hashlib.sha256(token_string.encode()).hexdigest()
        is_valid = token == expected_token
        print(f"Token validation - match: {is_valid}")
        print(f"Expected token: {expected_token}")
        print(f"Provided token: {token}")
        return is_valid
    except Exception as e:
        print(f"Token validation error: {str(e)}")
        return False

# Check authentication at the beginning
user, token, timestamp = get_url_params()
is_authenticated = validate_token(user, token, timestamp)

if not is_authenticated:
    st.error("⚠️ Authentication required. Please access this application through the main portal.")
    st.stop()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DOCUMENTS_DIR = "documents" # This might still be used or can be removed if agent.py handles all doc locations
# PROCESSED_DOCS_FILE = "processed_docs.json" # Removed
UPLOAD_EXTENSIONS = [".pdf", ".docx", ".doc", ".txt"] # May be removed if upload is fully removed
EMBEDDINGS_DIM = 4096 # Potentially unused
MAX_RETRIES = 3 # Potentially unused
BASE_TIMEOUT = 2400  # 40 minutes # Potentially unused

# Configure Streamlit page
st.set_page_config(
    page_title="Gyaan AI",
    page_icon="🔆",
    layout="wide",
    initial_sidebar_state="expanded"
)

ASSETS_DIR = "assets"

# Custom CSS for improved styling
st.markdown("""
    <style>
    /* Main container */
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Main content */
    .main-content {
        margin-bottom: 80px;
        padding: 20px;
    }
    
    /* Chat container */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
        margin-bottom: 100px;
    }
    
    /* Message styling */
    .chat-message {
        padding: 1.2rem;
        border-radius: 0.8rem;
        margin-bottom: 1.2rem;
        max-width: 85%;
    }
    
    .user-message {
        background-color: #F0F7FF;
        border-left: 4px solid #3B82F6;
        margin-left: auto;
    }
    
    .assistant-message {
        background-color: #F0FDF4;
        border-left: 4px solid #10B981;
        margin-right: auto;
    }
    
             
            
    /* Sidebar styling */
    .sidebar-header {
        font-size: 1.6em;
        font-weight: bold;
        padding: 15px 0;
        text-align: center;
        color: #1E293B;
        border-bottom: 2px solid #E2E8F0;
    }
    
    /* Document section */
    .doc-section {
        background-color: #F8FAFC;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
    }
    
    /* Upload section */
    .upload-section {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        border: 2px dashed #CBD5E1;
        text-align: center;
    }
    
    /* Footer */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        height: 50px;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        display: flex;
        align-items: center;
        justify-content: center;
        border-top: 1px solid #E2E8F0;
        z-index: 100;
    }
    
    /* Status messages */
    .status-message {
        padding: 12px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .success-message {
        background-color: #F0FDF4;
        color: #166534;
        border: 1px solid #BBF7D0;
    }
    
    .error-message {
        background-color: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FECACA;
    }
    
    /* Radio button styling */
    .stRadio > label {
        font-weight: 500;
        color: #334155;
    }
    
    /* Chat input container */
    .stChatInput {
        bottom: 60px !important;
        background: white;
        padding: 10px;
        border-top: 1px solid #E2E8F0;
    }
    
    /* Document title */
    .doc-title {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #3B82F6;
    }
    </style>
""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    """Initialize Streamlit session state variables."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = {}
    # if 'rag_instances' not in st.session_state: # Removed
    #     st.session_state.rag_instances = {} # Removed
    if 'selected_doc' not in st.session_state:
        st.session_state.selected_doc = None
    # if 'processing_status' not in st.session_state: # Removed - tied to old processing
    #     st.session_state.processing_status = {} # Removed

def initialize_directories():
    """Create necessary directories if they don't exist."""
    try:
        os.makedirs(DOCUMENTS_DIR, exist_ok=True) # Keep if DOCUMENTS_DIR is still relevant
        os.makedirs("pdfs", exist_ok=True) # Ensure pdfs directory exists for uploads
        # if not os.path.exists(PROCESSED_DOCS_FILE): # Removed
        #     with open(PROCESSED_DOCS_FILE, \'w\') as f: # Removed
        #         json.dump([], f) # Removed
        logger.info("Directories initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing directories: {str(e)}")
        raise

def get_user_stats():
    """Get statistics for the platform: total answers, total questions, and processed documents."""
    processed_docs_count = 0 # Simplified as PROCESSED_DOCS_FILE is removed
    # if os.path.exists(PROCESSED_DOCS_FILE): # Removed
    #     try: # Removed
    #         with open(PROCESSED_DOCS_FILE, 'r') as f: # Removed
    #             docs = json.load(f) # Removed
    #         processed_docs_count = len(docs) # Removed
    #     except Exception as e: # Removed
    #         logger.error(f"Error reading processed docs file for stats: {str(e)}") # Removed
    #         pass  # Continue if stats can't be loaded # Removed

    total_questions = 0
    total_answers = 0
    if 'chat_history' in st.session_state:
        for doc_name, history in st.session_state.chat_history.items():
            for message in history:
                if message.get("role") == "user":
                    total_questions += 1
                elif message.get("role") == "assistant":
                    total_answers += 1
    
    # Returns in the order they will appear on the chart: Answers, Questions, Documents
    return total_answers, total_questions, processed_docs_count

def create_stats_chart():
    """Create a bar chart with platform statistics."""
    total_answers, total_questions, processed_docs_count = get_user_stats()

    categories = ['Answers', 'Questions', 'Documents']
    counts = [total_answers, total_questions, processed_docs_count]
    
    # Color scheme: Answers (Red), Questions (Green), Documents (Blue)
    color_domain = ['Answers', 'Questions', 'Documents']
    color_range = ['#C44E52', '#55A868', '#4C72B0']

    data = pd.DataFrame({
        'Category': categories,
        'Count': counts
    })

    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X('Category', title=None, sort=None),  # sort=None to maintain DataFrame order
        y=alt.Y('Count', title=None, axis=alt.Axis(format='d')),
        color=alt.Color('Category', legend=None,
                      scale=alt.Scale(domain=color_domain, range=color_range)),
        tooltip=['Category', 'Count']
    ).properties(
        title='Platform Statistics',
        height=200
    ).configure_title(
        fontSize=14,
        anchor='middle'
    )
    return chart

# def load_processed_docs(): # Removed
#     """Load the list of processed documents.""" # Removed
#     try: # Removed
#         if os.path.exists(PROCESSED_DOCS_FILE): # Removed
#             with open(PROCESSED_DOCS_FILE, 'r') as f: # Removed
#                 docs = json.load(f) # Removed
#             logger.info(f"Loaded {len(docs)} processed documents") # Removed
#             return docs # Removed
#         return [] # Removed
#     except Exception as e: # Removed
#         logger.error(f"Error loading processed documents: {str(e)}") # Removed
#         return [] # Removed

# def save_processed_doc(doc_info): # Removed
#     """Save information about a processed document.""" # Removed
#     try: # Removed
#         docs = load_processed_docs() # Removed
#         if any(doc['name'] == doc_info['name'] for doc in docs): # Removed
#             logger.warning(f"Document {doc_info['name']} already exists") # Removed
#             return False # Removed
#         docs.append(doc_info) # Removed
#         with open(PROCESSED_DOCS_FILE, 'w') as f: # Removed
#             json.dump(docs, f) # Removed
#         logger.info(f"Document {doc_info['name']} saved successfully") # Removed
#         return True # Removed
#     except Exception as e: # Removed
#         logger.error(f"Error saving document info: {str(e)}") # Removed
#         raise # Removed


# async def process_document_with_retry(rag, text, progress_bar, status_placeholder): # Removed
#     """Process document with retry logic and increasing timeouts.""" # Removed
#     for attempt in range(MAX_RETRIES): # Removed
#         timeout = BASE_TIMEOUT * (attempt + 1) # Removed
#         try: # Removed
#             print(f"\\n=== Processing Attempt {attempt + 1}/{MAX_RETRIES} ===") # Removed
#             print(f"Timeout: {timeout} seconds") # Removed
            
#             status_placeholder.info(f"Processing attempt {attempt + 1}/{MAX_RETRIES} (timeout: {timeout}s)") # Removed
#             result = await asyncio.wait_for( # Removed
#                 rag.ainsert(text), # Removed
#                 timeout=timeout # Removed
#             ) # Removed
#             logger.info("Document processed successfully") # Removed
#             print("Document processing successful") # Removed
#             return result # Removed
#         except asyncio.TimeoutError: # Removed
#             if attempt < MAX_RETRIES - 1: # Removed
#                 logger.warning(f"Processing timeout on attempt {attempt + 1}") # Removed
#                 print(f"Timeout occurred - Retrying...") # Removed
#                 status_placeholder.warning(f"Processing timeout, attempt {attempt + 1} of {MAX_RETRIES}. Increasing timeout and retrying...") # Removed
#                 await asyncio.sleep(5) # Removed
#             else: # Removed
#                 error_msg = "Document processing failed after maximum retries" # Removed
#                 logger.error(error_msg) # Removed
#                 print(f"=== Error ===\\n{error_msg}") # Removed
#                 raise Exception(error_msg) # Removed
#         except Exception as e: # Removed
#             if attempt < MAX_RETRIES - 1: # Removed
#                 logger.warning(f"Processing error on attempt {attempt + 1}: {str(e)}") # Removed
#                 print(f"Error occurred: {str(e)} - Retrying...") # Removed
#                 status_placeholder.warning(f"Processing error: {str(e)}. Retrying...") # Removed
#                 await asyncio.sleep(5) # Removed
#             else: # Removed
#                 error_msg = f"Document processing failed after maximum retries: {str(e)}" # Removed
#                 logger.error(error_msg) # Removed
#                 print(f"=== Error ===\\n{error_msg}") # Removed
#                 raise Exception(error_msg) # Removed

# async def process_document(uploaded_file, progress_bar, status_placeholder): # Removed
#     """Process an uploaded document and create LightRAG instance.""" # Removed
#     try: # Removed
#         logger.info(f"Starting to process document: {uploaded_file.name}") # Removed
#         print(f"\\n=== Processing Document ===\\nFile: {uploaded_file.name}") # Removed
        
#         # Update progress # Removed
#         progress_bar.progress(10, text="Reading file...") # Removed
        
#         # Read file content # Removed
#         content = uploaded_file.read() # Removed
#         logger.info(f"File read successfully, size: {len(content)} bytes") # Removed
#         print(f"File read successfully - Size: {len(content)} bytes") # Removed
        
#         # Process document # Removed
#         progress_bar.progress(20, text="Extracting text...") # Removed
#         # doc_processor = DocumentProcessor() # This was the source of the error, DocumentProcessor is not defined # Removed
#         # extracted_text = doc_processor.process_document(content, uploaded_file.name) # Removed
#         extracted_text = "Dummy text, processing removed" # Placeholder if needed, but function is removed # Removed
#         logger.info(f"Text extracted successfully, length: {len(extracted_text)}") # Removed
#         print(f"Text extracted - Length: {len(extracted_text)} characters") # Removed
        
#         # Save processed document # Removed
#         doc_path = os.path.join(DOCUMENTS_DIR, uploaded_file.name) # Removed
#         with open(doc_path, 'wb') as f: # Removed
#             f.write(content) # Removed
        
#         # Save text content # Removed
#         text_path = os.path.join(DOCUMENTS_DIR, f"{uploaded_file.name}.txt") # Removed
#         async with aiofiles.open(text_path, mode='w') as f: # Removed
#             await f.write(extracted_text) # Removed
#         logger.info(f"Document saved to: {doc_path}") # Removed
#         print(f"Document saved to: {doc_path}") # Removed
        
#         # Initialize LightRAG # Removed
#         progress_bar.progress(40, text="Initializing LightRAG...") # Removed
#         doc_dir = os.path.join(DOCUMENTS_DIR, f"{uploaded_file.name}_rag") # Removed
#         os.makedirs(doc_dir, exist_ok=True) # Removed
#         rag = initialize_lightrag(doc_dir) # Removed
        
#         # Process document with LightRAG # Removed
#         progress_bar.progress(60, text="Processing with LightRAG...") # Removed
#         status_placeholder.info("Starting document processing with LightRAG. This may take several minutes...") # Removed
        
#         await process_document_with_retry(rag, extracted_text, progress_bar, status_placeholder) # Removed
        
#         # Save document info # Removed
#         progress_bar.progress(80, text="Saving document information...") # Removed
#         doc_info = { # Removed
#             "name": uploaded_file.name, # Removed
#             "path": doc_path, # Removed
#             "text_path": text_path, # Removed
#             "rag_dir": doc_dir, # Removed
#             "processed_date": datetime.now().isoformat() # Removed
#         } # Removed
        
#         if not save_processed_doc(doc_info): # Removed
#             return False, "Document already exists!" # Removed
        
#         # Store RAG instance # Removed
#         st.session_state.rag_instances[uploaded_file.name] = rag # Removed
        
#         progress_bar.progress(100, text="Complete!") # Removed
#         logger.info(f"Document {uploaded_file.name} processed successfully") # Removed
#         print("Document processing completed successfully") # Removed
#         return True, "Document processed successfully!" # Removed
        
#     except Exception as e: # Removed
#         error_msg = f"Error processing document: {str(e)}" # Removed
#         logger.error(error_msg) # Removed
#         print(f"\\n=== Error ===\\n{error_msg}") # Removed
#         return False, error_msg # Removed

# def initialize_lightrag(working_dir): # Removed
#     """Initialize LightRAG with optimized parameters""" # Removed
#     try: # Removed
#         logger.info(f"Initializing LightRAG in directory: {working_dir}") # Removed
#         print(f"\\n=== Initializing LightRAG ===\\nWorking Directory: {working_dir}") # Removed
        
#         # custom_embed = CustomEmbedding(timeout=BASE_TIMEOUT) # Removed - CustomEmbedding not defined
#         # custom_embed.embedding_dim = EMBEDDINGS_DIM  # Ensure embedding_dim is accessible # Removed
        
#         # rag = LightRAG( # Removed - LightRAG not defined
#         #     working_dir=working_dir, # Removed
#         #     llm_model_func=get_full_response, # Removed - get_full_response not defined
#         #     embedding_func=custom_embed, # Removed
#         #     llm_model_name="ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4", # Removed
#         #     node2vec_params={ # Removed
#         #         "dimensions": EMBEDDINGS_DIM, # Removed
#         #         "num_walks": 15, # Removed
#         #         "walk_length": 50, # Removed
#         #         "window_size": 3, # Removed
#         #         "iterations": 4, # Removed
#         #         "random_seed": 3, # Removed
#         #     }, # Removed
#         #     vector_db_storage_cls_kwargs={ # Removed
#         #         "embedding_dim": EMBEDDINGS_DIM, # Removed
#         #     }, # Removed
#         # ) # Removed
#         logger.info("LightRAG initialized successfully (stubbed)") # Removed
#         print("LightRAG initialization successful (stubbed)") # Removed
#         # return rag # Removed
#         return None # Placeholder since function is removed # Removed
#     except Exception as e: # Removed
#         error_msg = f"Error initializing LightRAG: {str(e)}" # Removed
#         logger.error(error_msg) # Removed
#         print(f"=== Error ===\\n{error_msg}") # Removed
#         raise # Removed


# async def generate_response(prompt, selected_doc, conversation_history=None): # Removed
#     """Generate response using enhanced context retrieval""" # Removed
#     try: # Removed
#         logger.info(f"Generating response for document: {selected_doc['name']} (stubbed)") # Removed
#         # This function relies on LightRAG and related components which are being removed. # Removed
#         # For now, it will return a placeholder response. # Removed
#         # You will need to integrate with agent.py for actual responses. # Removed
#         await asyncio.sleep(1) # Simulate async work # Removed
#         return f"Response generation for '{prompt}' on doc '{selected_doc['name']}' is not implemented in gyaan_doc.py anymore." # Removed
        
#     except Exception as e: # Removed
#         error_msg = f"Error generating response: {str(e)}" # Removed
#         logger.error(error_msg) # Removed
#         print(f"\\n=== Error ===\\n{error_msg}") # Removed
#         return error_msg # Removed
def main():
    # Initialize
    init_session_state()
    initialize_directories()
    
    # Add your CSS here
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Platform Statistics")
        try:
            stats_chart = create_stats_chart()
            st.altair_chart(stats_chart, use_container_width=True)
        except Exception as e:
            st.error(f"Could not display statistics: {str(e)}")
        st.markdown("---") # Add a visual separator

        st.markdown("### 📚 Document Manager")
        
        # Upload section
        st.markdown("### 📥 Upload Document")
        uploaded_files = st.file_uploader(
            "Upload PDF or DOCX files",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            help="Upload one or more PDF or DOCX files. These will be placed in the 'pdfs' directory for the agent to process."
        )

        if uploaded_files:
            save_dir = "pdfs"
            for uploaded_file in uploaded_files:
                file_path = os.path.join(save_dir, uploaded_file.name)
                try:
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.success(f"Successfully uploaded and saved {uploaded_file.name} to {save_dir}/")
                except Exception as e:
                    st.error(f"Error saving {uploaded_file.name}: {e}")
            st.info("The agent will process newly uploaded files based on its configuration (e.g., on restart or next scheduled scan).")

        # Show processed documents - Simplified
        st.markdown("### 📚 Available Documents")
        # The agent handles document interactions.
        # You might want to add a way to list files in the 'pdfs' folder here if desired,
        # but for now, we'll keep it simple as per the removal of the previous message.
        st.caption("Uploaded documents will be processed by the agent.")
    
    # # Main interface
    # st.title("🔆 Gyaan AI")

    try:
        logo_path = os.path.join(ASSETS_DIR, "GYaan_logo.jpeg")
        gyaan_logo = Image.open(logo_path)
        st.image(gyaan_logo, width=300)
    except Exception as e:
        st.error(f"Could not load Gyaan logo: {str(e)}")
        st.title("Gyaan AI")  # Fallback to text if image fails to load

    
    if not st.session_state.selected_doc:
        st.info("👈 Please interact with the agent for document-related queries.") # Updated message
        
        
    else:
        # Display current document
        st.markdown(f"""
            ### 📄 Current Document: {st.session_state.selected_doc['name']}
        """)
        
        # Initialize chat history for current document
        current_doc = st.session_state.selected_doc['name']
        if current_doc not in st.session_state.chat_history:
            st.session_state.chat_history[current_doc] = []
        
        # Display chat messages
        for message in st.session_state.chat_history[current_doc]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about the document..."):
            # Add user message
            st.session_state.chat_history[current_doc].append({
                "role": "user",
                "content": prompt
            })
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and display response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing document and generating response..."):
                    # response = asyncio.run( # Call to generate_response is removed
                    #     generate_response( 
                    #         prompt, 
                    #         st.session_state.selected_doc,
                    #         st.session_state.chat_history[current_doc]
                    #     )
                    # )
                    response = "Response generation has been moved to the agent." # Placeholder
                    st.markdown(response)
                    st.session_state.chat_history[current_doc].append({
                        "role": "assistant",
                        "content": response
                    })
    
    # # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    try:
        with col1:
            ursc_logo_path = os.path.join(ASSETS_DIR, "ursc_light.png")
            ursc_logo = Image.open(ursc_logo_path)
            st.image(ursc_logo, width=100)
        
        with col2:
            st.markdown("<p style='text-align: center; margin-top: 20px;'>Built by Team GYAAN</p>", unsafe_allow_html=True)
        
        with col3:
            isro_logo_path = os.path.join(ASSETS_DIR, "ISROLogo.png")
            isro_logo = Image.open(isro_logo_path)
            st.image(isro_logo, width=100)
    except Exception as e:
        st.error(f"Could not load footer logos: {str(e)}")
        # Fallback to simple text footer
        st.markdown("<p style='text-align: center;'>Built by Team GYAAN</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()