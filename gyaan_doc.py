import streamlit as st
import os
from pathlib import Path
import asyncio
import json
import aiofiles
from datetime import datetime
import time
import logging
# from document_processor import DocumentProcessor
# from custom_embeddings import CustomEmbedding
# from custom_llm import get_full_response
# from lightrag import LightRAG, QueryParam
# from context_manager import ContextManager
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
DOCUMENTS_DIR = "documents"
PROCESSED_DOCS_FILE = "processed_docs.json"
UPLOAD_EXTENSIONS = [".pdf", ".docx", ".doc", ".txt"]
EMBEDDINGS_DIM = 4096
MAX_RETRIES = 3
BASE_TIMEOUT = 2400  # 40 minutes

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
    if 'rag_instances' not in st.session_state:
        st.session_state.rag_instances = {}
    if 'selected_doc' not in st.session_state:
        st.session_state.selected_doc = None
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = {}

def initialize_directories():
    """Create necessary directories if they don't exist."""
    try:
        os.makedirs(DOCUMENTS_DIR, exist_ok=True)
        if not os.path.exists(PROCESSED_DOCS_FILE):
            with open(PROCESSED_DOCS_FILE, 'w') as f:
                json.dump([], f)
        logger.info("Directories initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing directories: {str(e)}")
        raise

def get_user_stats():
    """Get statistics for the platform: total answers, total questions, and processed documents."""
    processed_docs_count = 0
    if os.path.exists(PROCESSED_DOCS_FILE):
        try:
            with open(PROCESSED_DOCS_FILE, 'r') as f:
                docs = json.load(f)
            processed_docs_count = len(docs)
        except Exception as e:
            logger.error(f"Error reading processed docs file for stats: {str(e)}")
            pass  # Continue if stats can't be loaded

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

def load_processed_docs():
    """Load the list of processed documents."""
    try:
        if os.path.exists(PROCESSED_DOCS_FILE):
            with open(PROCESSED_DOCS_FILE, 'r') as f:
                docs = json.load(f)
            logger.info(f"Loaded {len(docs)} processed documents")
            return docs
        return []
    except Exception as e:
        logger.error(f"Error loading processed documents: {str(e)}")
        return []

def save_processed_doc(doc_info):
    """Save information about a processed document."""
    try:
        docs = load_processed_docs()
        if any(doc['name'] == doc_info['name'] for doc in docs):
            logger.warning(f"Document {doc_info['name']} already exists")
            return False
        docs.append(doc_info)
        with open(PROCESSED_DOCS_FILE, 'w') as f:
            json.dump(docs, f)
        logger.info(f"Document {doc_info['name']} saved successfully")
        return True
    except Exception as e:
        logger.error(f"Error saving document info: {str(e)}")
        raise


async def process_document_with_retry(rag, text, progress_bar, status_placeholder):
    """Process document with retry logic and increasing timeouts."""
    for attempt in range(MAX_RETRIES):
        timeout = BASE_TIMEOUT * (attempt + 1)
        try:
            print(f"\n=== Processing Attempt {attempt + 1}/{MAX_RETRIES} ===")
            print(f"Timeout: {timeout} seconds")
            
            status_placeholder.info(f"Processing attempt {attempt + 1}/{MAX_RETRIES} (timeout: {timeout}s)")
            result = await asyncio.wait_for(
                rag.ainsert(text),
                timeout=timeout
            )
            logger.info("Document processed successfully")
            print("Document processing successful")
            return result
        except asyncio.TimeoutError:
            if attempt < MAX_RETRIES - 1:
                logger.warning(f"Processing timeout on attempt {attempt + 1}")
                print(f"Timeout occurred - Retrying...")
                status_placeholder.warning(f"Processing timeout, attempt {attempt + 1} of {MAX_RETRIES}. Increasing timeout and retrying...")
                await asyncio.sleep(5)
            else:
                error_msg = "Document processing failed after maximum retries"
                logger.error(error_msg)
                print(f"=== Error ===\n{error_msg}")
                raise Exception(error_msg)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                logger.warning(f"Processing error on attempt {attempt + 1}: {str(e)}")
                print(f"Error occurred: {str(e)} - Retrying...")
                status_placeholder.warning(f"Processing error: {str(e)}. Retrying...")
                await asyncio.sleep(5)
            else:
                error_msg = f"Document processing failed after maximum retries: {str(e)}"
                logger.error(error_msg)
                print(f"=== Error ===\n{error_msg}")
                raise Exception(error_msg)

async def process_document(uploaded_file, progress_bar, status_placeholder):
    """Process an uploaded document and create LightRAG instance."""
    try:
        logger.info(f"Starting to process document: {uploaded_file.name}")
        print(f"\n=== Processing Document ===\nFile: {uploaded_file.name}")
        
        # Update progress
        progress_bar.progress(10, text="Reading file...")
        
        # Read file content
        content = uploaded_file.read()
        logger.info(f"File read successfully, size: {len(content)} bytes")
        print(f"File read successfully - Size: {len(content)} bytes")
        
        # Process document
        progress_bar.progress(20, text="Extracting text...")
        doc_processor = DocumentProcessor()
        extracted_text = doc_processor.process_document(content, uploaded_file.name)
        logger.info(f"Text extracted successfully, length: {len(extracted_text)}")
        print(f"Text extracted - Length: {len(extracted_text)} characters")
        
        # Save processed document
        doc_path = os.path.join(DOCUMENTS_DIR, uploaded_file.name)
        with open(doc_path, 'wb') as f:
            f.write(content)
        
        # Save text content
        text_path = os.path.join(DOCUMENTS_DIR, f"{uploaded_file.name}.txt")
        async with aiofiles.open(text_path, mode='w') as f:
            await f.write(extracted_text)
        logger.info(f"Document saved to: {doc_path}")
        print(f"Document saved to: {doc_path}")
        
        # Initialize LightRAG
        progress_bar.progress(40, text="Initializing LightRAG...")
        doc_dir = os.path.join(DOCUMENTS_DIR, f"{uploaded_file.name}_rag")
        os.makedirs(doc_dir, exist_ok=True)
        rag = initialize_lightrag(doc_dir)
        
        # Process document with LightRAG
        progress_bar.progress(60, text="Processing with LightRAG...")
        status_placeholder.info("Starting document processing with LightRAG. This may take several minutes...")
        
        await process_document_with_retry(rag, extracted_text, progress_bar, status_placeholder)
        
        # Save document info
        progress_bar.progress(80, text="Saving document information...")
        doc_info = {
            "name": uploaded_file.name,
            "path": doc_path,
            "text_path": text_path,
            "rag_dir": doc_dir,
            "processed_date": datetime.now().isoformat()
        }
        
        if not save_processed_doc(doc_info):
            return False, "Document already exists!"
        
        # Store RAG instance
        st.session_state.rag_instances[uploaded_file.name] = rag
        
        progress_bar.progress(100, text="Complete!")
        logger.info(f"Document {uploaded_file.name} processed successfully")
        print("Document processing completed successfully")
        return True, "Document processed successfully!"
        
    except Exception as e:
        error_msg = f"Error processing document: {str(e)}"
        logger.error(error_msg)
        print(f"\n=== Error ===\n{error_msg}")
        return False, error_msg

def initialize_lightrag(working_dir):
    """Initialize LightRAG with optimized parameters"""
    try:
        logger.info(f"Initializing LightRAG in directory: {working_dir}")
        print(f"\n=== Initializing LightRAG ===\nWorking Directory: {working_dir}")
        
        custom_embed = CustomEmbedding(timeout=BASE_TIMEOUT)
        custom_embed.embedding_dim = EMBEDDINGS_DIM  # Ensure embedding_dim is accessible
        
        rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=get_full_response,
            embedding_func=custom_embed,
            llm_model_name="ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4",
            node2vec_params={
                "dimensions": EMBEDDINGS_DIM,
                "num_walks": 15,
                "walk_length": 50,
                "window_size": 3,
                "iterations": 4,
                "random_seed": 3,
            },
            vector_db_storage_cls_kwargs={
                "embedding_dim": EMBEDDINGS_DIM,
            },
        )
        logger.info("LightRAG initialized successfully")
        print("LightRAG initialization successful")
        return rag
    except Exception as e:
        error_msg = f"Error initializing LightRAG: {str(e)}"
        logger.error(error_msg)
        print(f"=== Error ===\n{error_msg}")
        raise
        logger.info("LightRAG initialized successfully")
        print("LightRAG initialization successful")
        return rag
    except Exception as e:
        error_msg = f"Error initializing LightRAG: {str(e)}"
        logger.error(error_msg)
        print(f"=== Error ===\n{error_msg}")
        raise
        logger.info("LightRAG initialized successfully")
        print("LightRAG initialization successful")
        return rag
    except Exception as e:
        error_msg = f"Error initializing LightRAG: {str(e)}"
        logger.error(error_msg)
        print(f"=== Error ===\n{error_msg}")
        raise

async def generate_response(prompt, selected_doc, conversation_history=None):
    """Generate response using enhanced context retrieval"""
    try:
        logger.info(f"Generating response for document: {selected_doc['name']}")
        context_manager = ContextManager(max_tokens=4096)
        
        # Initialize/get RAG instance
        if selected_doc["name"] not in st.session_state.rag_instances:
            logger.info("Initializing new RAG instance")
            rag = initialize_lightrag(selected_doc["rag_dir"])
            st.session_state.rag_instances[selected_doc["name"]] = rag
        else:
            logger.info("Using existing RAG instance")
            rag = st.session_state.rag_instances[selected_doc["name"]]

        # Optimize query parameters
        context_param = QueryParam(
            mode="hybrid",
            top_k=60,                    # Reduced from 10 to 5
            max_token_for_text_unit=4000,
            max_token_for_global_context=2000,
            max_token_for_local_context=2000,
            only_need_context=True
        )
        
        # Get context from LightRAG
        context_response = await rag.aquery(prompt, param=context_param)
        
        # Process context sections
        context_sections = []
        
        # 1. Add document metadata
        context_sections.append(f"Document: {selected_doc['name']}")
        
        # 2. Process text content from context response
        if isinstance(context_response, str):
            # Split the content on ----- markers
            content_parts = context_response.split('-----')
            
            # Look for content part
            text_content = None
            for part in content_parts:
                if 'Content' in part:
                    text_content = part.replace('Content', '').strip()
                    break
            
            if text_content:
                # Split into manageable chunks
                chunks = text_content.split('\n\n')
                chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
                
                # Get top 10 most relevant chunks
                processed_chunks = []
                for i, chunk in enumerate(chunks[:10], 1):
                    processed_chunks.append(f"[Excerpt {i}]\n{chunk}")
                
                # Truncate chunks to stay within token limits
                truncated_chunks = context_manager.truncate_context(processed_chunks)
                if truncated_chunks:
                    context_sections.append("Relevant Document Sections:\n" + "\n\n".join(truncated_chunks))
        
        # 3. Add entities and relationships
        er_context = context_manager.extract_entities_relationships(context_response)
        if er_context:
            context_sections.append(er_context)
        
        # Combine all context
        context_text = "\n\n".join(context_sections)
        
        # Create augmented prompt
        augmented_prompt = context_manager.create_augmented_prompt(
            selected_doc['name'],
            context_text,
            prompt
        )
        
        # Log context and prompt for debugging
        logger.info(f"Context Length: {len(context_text)}")
        logger.debug(f"Augmented Prompt: {augmented_prompt}")
        
        # Get response
        response = await get_full_response(augmented_prompt, max_tokens=4096)
        return response
        
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        logger.error(error_msg)
        print(f"\n=== Error ===\n{error_msg}")
        return error_msg
   

# async def generate_response(prompt, selected_doc, conversation_history=None):
#     try:
#         logger.info(f"Generating response for document: {selected_doc['name']}")
        
#         # Initialize/get RAG instance
#         if selected_doc["name"] not in st.session_state.rag_instances:
#             logger.info("Initializing new RAG instance")
#             rag = initialize_lightrag(selected_doc["rag_dir"])
#             st.session_state.rag_instances[selected_doc["name"]] = rag
#         else:
#             logger.info("Using existing RAG instance")
#             rag = st.session_state.rag_instances[selected_doc["name"]]

#         # Get initial context
#         context_param = QueryParam(
#             mode="hybrid",
#             top_k=10,
#             max_token_for_text_unit=2000,
#             max_token_for_global_context=2000,
#             max_token_for_local_context=2000,
#             only_need_context=True
#         )
        
#         # Get context from LightRAG
#         context_response = await rag.aquery(prompt, param=context_param)
#         print("\n=== Initial Context ===")
#         print(context_response)
        
#         # Process and filter context
#         context_parts = []
        
#         # Handle string response
#         if isinstance(context_response, str):
#             # Extract entities
#             if '-----Entities-----' in context_response:
#                 entities_text = context_response.split('-----Entities-----')[1].split('-----Relationships-----')[0]
#                 entities = []
                
#                 for line in entities_text.split('\n'):
#                     if '"' in line and ',' in line and not line.startswith('id'):
#                         try:
#                             # Split by quotes to get entity and description
#                             parts = line.split('"')
#                             if len(parts) >= 5:
#                                 entity = parts[1]
#                                 description = parts[3]
#                                 # Get rank from last comma-separated value
#                                 rank = float(line.split(',')[-1].strip())
#                                 entities.append((rank, f"- {entity}: {description}"))
#                         except:
#                             continue
                
#                 # Add top 3 entities
#                 if entities:
#                     entities.sort(key=lambda x: x[0], reverse=True)
#                     top_entities = [e[1] for e in entities[:20]]
#                     context_parts.append("Key Concepts:\n" + "\n".join(top_entities))
            
#             # Extract relationships
#             if '-----Relationships-----' in context_response:
#                 relations_text = context_response.split('-----Relationships-----')[1]
#                 if '-----' in relations_text:
#                     relations_text = relations_text.split('-----')[0]
#                 relations = []
                
#                 for line in relations_text.split('\n'):
#                     if '"' in line and ',' in line and not line.startswith('id'):
#                         try:
#                             # Split by quotes to get description
#                             parts = line.split('"')
#                             if len(parts) >= 6:
#                                 description = parts[5]
#                                 # Get rank (third from last comma-separated value)
#                                 comma_parts = line.split(',')
#                                 rank = float(comma_parts[-3].strip())
#                                 relations.append((rank, f"- {description}"))
#                         except:
#                             continue
                
#                 # Add top 2 relationships
#                 if relations:
#                     relations.sort(key=lambda x: x[0], reverse=True)
#                     top_relations = [r[1] for r in relations[:20]]
#                     context_parts.append("Key Findings:\n" + "\n".join(top_relations))
        
#         # Combine the filtered context
#         context_text = "\n\n".join(context_parts)
        
#         print("\n=== Final Context ===")
#         print(f"Context Length: {len(context_text)}")
#         print("Context Preview:")
#         print(context_text[:500])
        
#         # Create augmented prompt
#         augmented_prompt = f"""I am analyzing the document {selected_doc['name']}. Here is the key information from the document:

# {context_text}

# Based only on the information provided above, please {prompt}. If certain aspects are not covered in this context, please mention that explicitly."""

#         print("\n=== Final Augmented Prompt ===")
#         print(augmented_prompt)

#         # Get response
#         response = await get_full_response(augmented_prompt, max_tokens=4096, kwargs={})
#         return response
        
#     except Exception as e:
#         error_msg = f"Error generating response: {str(e)}"
#         logger.error(error_msg)
#         print(f"\n=== Error ===\n{error_msg}")
#         return error_msg

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
        uploaded_file = st.file_uploader(
            "Upload a document",
            type=[ext.replace(".", "") for ext in UPLOAD_EXTENSIONS],
            help="Supported formats: PDF, DOCX, DOC, TXT"
        )
        
        # Process uploaded file
        if uploaded_file is not None:
            if any(doc['name'] == uploaded_file.name for doc in load_processed_docs()):
                st.error(f"Document '{uploaded_file.name}' already exists!")
            else:
                try:
                    progress_placeholder = st.empty()
                    progress_bar = progress_placeholder.progress(0)
                    status_placeholder = st.empty()
                    
                    with st.spinner("Processing document..."):
                        success, message = asyncio.run(
                            process_document(uploaded_file, progress_bar, status_placeholder)
                        )
                        
                        progress_placeholder.empty()
                        
                        if success:
                            status_placeholder.success(message)
                            time.sleep(1)
                            st.rerun()
                        else:
                            status_placeholder.error(message)
                except Exception as e:
                    logger.error(f"Document processing error: {str(e)}")
                    st.error(f"An error occurred: {str(e)}")
        
        # Show processed documents
        st.markdown("### 📚 Available Documents")
        processed_docs = load_processed_docs()
        
        if not processed_docs:
            st.info("No documents available. Please upload a document to begin.")
        else:
            # Radio button selection for documents
            doc_options = ["None"] + [doc["name"] for doc in processed_docs]
            selected_doc_name = st.radio(
                "Select a document to chat with:",
                options=doc_options,
                index=0,
                format_func=lambda x: x if x == "None" else f"📄 {x}"
            )
            
            if selected_doc_name != "None":
                st.session_state.selected_doc = next(
                    (doc for doc in processed_docs if doc["name"] == selected_doc_name),
                    None
                )
            else:
                st.session_state.selected_doc = None
    
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
        st.info("👈 Please select a document from the sidebar to start chatting.")
        
        
        
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
                    response = asyncio.run(
                        generate_response(
                            prompt, 
                            st.session_state.selected_doc,
                            st.session_state.chat_history[current_doc]
                        )
                    )
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