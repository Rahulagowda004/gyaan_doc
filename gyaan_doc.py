import streamlit as st
import os
import logging
from PIL import Image
import uuid
import json
import time
from datetime import datetime
from agent import app as agent_app
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from typing import Sequence

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
PDFS_DIR = "pdfs" # Used for saving uploaded files
ASSETS_DIR = "assets"
CHAT_LOG_FILE = "chat_history.json"
STATS_FILE = "stats.json"  # Added constant for stats file

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
        st.session_state.chat_history = []
    if 'agent_thread_id' not in st.session_state:
        st.session_state.agent_thread_id = str(uuid.uuid4())
    if 'username' not in st.session_state:
        st.session_state.username = f"user_{uuid.uuid4().hex[:8]}"
        # Initialize retriever for new user
        from agent import reinitialize_retriever, set_tools_context
        set_tools_context(st.session_state.username)
        reinitialize_retriever(st.session_state.username)


# Create necessary directories
def initialize_directories():
    """Create necessary directories if they don't exist."""
    if not os.path.exists(PDFS_DIR):
        os.makedirs(PDFS_DIR)
    if not os.path.exists(ASSETS_DIR): # Ensure assets dir is also checked/created if needed by app
        os.makedirs(ASSETS_DIR)


def load_stats():
    """Load statistics from the JSON file."""
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.error("Error reading stats file. Starting with empty stats.")
    return {
        "total_interactions": 0,
        "total_users": 0,
        "unique_user_threads": [],
        "last_updated": datetime.now().isoformat()
    }

def save_stats(stats):
    """Save statistics to the JSON file."""
    stats["last_updated"] = datetime.now().isoformat()
    try:
        with open(STATS_FILE, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info("Statistics saved successfully")
    except Exception as e:
        logger.error(f"Error saving statistics: {str(e)}")

def update_user_stats():
    """Update user statistics with current session data."""
    stats = load_stats()
    
    # Count current session's new interactions
    if 'chat_history' in st.session_state:
        current_session_interactions = len([m for m in st.session_state.chat_history if m.get("role") == "user"])
        stats["total_interactions"] = max(stats["total_interactions"], 0) + current_session_interactions
    
    # Update unique users
    current_thread = st.session_state.get('agent_thread_id', None)
    if current_thread and current_thread not in stats["unique_user_threads"]:
        stats["unique_user_threads"].append(current_thread)
        stats["total_users"] = len(stats["unique_user_threads"])
    
    save_stats(stats)
    return stats["total_interactions"], stats["total_users"]

def get_user_stats():
    """Get statistics for the platform: total interactions and total users."""
    return update_user_stats()

# New function to interact with the LangGraph agent
def get_agent_response(current_chat_history: list[dict], thread_id: str) -> str:
    """
    Gets a response from the LangGraph agent using the current chat history and a thread_id.
    """
    messages_for_agent: list[BaseMessage] = []
    for entry in current_chat_history:
        content = entry["content"]
        if not isinstance(content, str): # Ensure content is string
            content = str(content)

        if entry["role"] == "user":
            messages_for_agent.append(HumanMessage(content=content))
        elif entry["role"] == "assistant":
            messages_for_agent.append(AIMessage(content=content))

    if not messages_for_agent or not isinstance(messages_for_agent[-1], HumanMessage):
        logger.error("Agent called without a final HumanMessage in history.")
        # This can happen if history is empty and a system tries to get a response.
        # For user-initiated chat, history will have at least one user message.
        return "Error: Agent requires a user prompt to respond."

    input_data = {"messages": messages_for_agent}
    # agent_config was imported from agent.py, but we need a session-specific thread_id
    current_agent_config = {"configurable": {"thread_id": thread_id}}
    
    final_ai_response = "Sorry, I couldn't get a response from the agent at this time."
    
    try:
        logger.info(f"Sending to agent (thread_id: {thread_id}): {len(messages_for_agent)} messages.")
        
        for event in agent_app.stream(input_data, current_agent_config, stream_mode="values"):
            if event and "messages" in event and event["messages"]:
                last_message = event["messages"][-1]
                if isinstance(last_message, AIMessage):
                    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
                        final_ai_response = last_message.content
                        logger.info(f"Agent final AIMessage content (thread_id: {thread_id}): {final_ai_response[:100]}...") # Log snippet
                    # else: # Intermediate AIMessage with tool_calls, stream continues
                        # logger.debug(f"Agent intermediate AIMessage with tool_calls (thread_id: {thread_id})")
        
        logger.info(f"Received from agent (thread_id: {thread_id}): {final_ai_response[:100]}...")
            
    except Exception as e:
        logger.error(f"Error interacting with LangGraph agent (thread_id: {thread_id}): {str(e)}", exc_info=True)
        final_ai_response = f"An error occurred while trying to reach the agent: {str(e)}"
        
    return final_ai_response

def load_chat_log():
    """Load the chat log from JSON file."""
    if os.path.exists(CHAT_LOG_FILE):
        try:
            with open(CHAT_LOG_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.error("Error reading chat log file. Starting with empty log.")
    return []

def save_chat_interaction(question: str, answer: str, time_taken: float):
    """Save a chat interaction to the JSON file."""
    chat_log = load_chat_log()
    
    interaction = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "time_taken_seconds": time_taken,
        "thread_id": st.session_state.get('agent_thread_id', 'unknown')
    }
    
    chat_log.append(interaction)
    
    try:
        with open(CHAT_LOG_FILE, 'w') as f:
            json.dump(chat_log, f, indent=2)
        logger.info(f"Saved chat interaction to {CHAT_LOG_FILE}")
    except Exception as e:
        logger.error(f"Error saving chat log: {str(e)}")

def get_user_docs_dir(username: str) -> str:
    """Get the user-specific documents directory."""
    user_dir = os.path.join(PDFS_DIR, username)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
    return user_dir

def create_temporary_success_message(message: str, container):
    """Create a success message that disappears after 3 seconds."""
    placeholder = container.empty()
    placeholder.success(message)
    time.sleep(3)
    placeholder.empty()

def handle_document_upload(uploaded_files, username: str):
    """Handle document upload for a specific user."""
    if not uploaded_files:
        return False

    user_docs_dir = get_user_docs_dir(username)
    files_saved = False
    
    # Create a container for temporary messages
    message_container = st.empty()
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(user_docs_dir, uploaded_file.name)
        try:
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            create_temporary_success_message(f"Successfully uploaded {uploaded_file.name}", message_container)
            files_saved = True
        except Exception as e:
            st.error(f"Error saving {uploaded_file.name}: {e}")
            logger.error(f"Error saving file for user {username}: {e}")
    
    if files_saved:
        with st.spinner("Processing new documents..."):
            try:
                from agent import reinitialize_retriever, set_tools_context
                set_tools_context(username)  # Set the context for tools
                reinitialize_retriever(username)
                create_temporary_success_message("Successfully processed new documents!", message_container)
            except Exception as e:
                st.error(f"Error processing documents: {e}")
                logger.error(f"Error processing documents for user {username}: {e}")
    
    return files_saved

def main():
    # Initialize
    init_session_state()
    initialize_directories()
    
    # Get or set username (you can modify this based on your authentication system)
    if 'username' not in st.session_state:
        st.session_state.username = f"user_{uuid.uuid4().hex[:8]}"
    
    # Set the tools context for the current user
    from agent import set_tools_context
    set_tools_context(st.session_state.username)
    
    # Sidebar
    with st.sidebar:
        # Platform Statistics as a dropdown (expander)
        with st.expander("📊 Platform Statistics", expanded=True):
            try:
                total_interactions, total_users = get_user_stats()
                st.markdown(f"**Total Interactions:** {total_interactions}")
                st.markdown(f"**Total Users:** {total_users}")
            except Exception as e:
                st.error(f"Could not display statistics: {str(e)}")
        
        st.markdown("---") # Add a visual separator

        st.markdown("### 📚 Document Manager")
        
        # Modified upload section
        st.markdown("### 📥 Upload Document")
        uploaded_files = st.file_uploader(
            "Upload PDF or DOCX files",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.username}",  # Unique key per user
            help="Upload one or more PDF or DOCX files"
        )

        if uploaded_files:
            handle_document_upload(uploaded_files, st.session_state.username)

        # Show available documents - User specific
        st.markdown("### 📚 Your Documents")
        user_docs = os.listdir(get_user_docs_dir(st.session_state.username))
        if user_docs:
            for doc in user_docs:
                st.text(f"📄 {doc}")
        else:
            st.info("No documents uploaded yet")

    # Main interface - Gyaan AI Logo
    try:
        logo_path = os.path.join(ASSETS_DIR, "GYaan_logo.jpeg")
        gyaan_logo = Image.open(logo_path)
        # Center the logo using columns
        col1_logo, col2_logo, col3_logo = st.columns([1,2,1])
        with col2_logo:
            st.image(gyaan_logo, width=300)
    except Exception as e:
        st.error(f"Could not load Gyaan logo: {str(e)}")
        st.title("Gyaan AI")  # Fallback to text if image fails to load

    # Main area (Chat Interface)
    # The conditional logic for st.session_state.selected_doc has been removed.
    # The chat interface is now always displayed.

    st.info("👈 Upload documents via the sidebar. Interact with the agent below.")

    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    # Display chat messages from history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"], avatar= "🧑‍💻" if message["role"] == "user" else "💡"):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        # Display user message
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)
        
        # Get and display assistant response
        with st.chat_message("assistant", avatar="💡"):
            with st.spinner("Thinking..."):
                # Start timing the response
                start_time = time.time()
                
                # Pass the entire chat history for context
                response_text = get_agent_response(st.session_state.chat_history, st.session_state.agent_thread_id)
                
                # Calculate time taken
                time_taken = time.time() - start_time
                  # Save the interaction to JSON
                save_chat_interaction(prompt, response_text, time_taken)
                
                # Update and save user statistics
                update_user_stats()
                
                st.markdown(response_text)
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})
    
    st.markdown("</div>", unsafe_allow_html=True) # End chat-container
    
    # Footer
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