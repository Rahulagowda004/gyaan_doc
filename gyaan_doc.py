import streamlit as st
import os
import logging
from PIL import Image
import uuid # Added for unique thread IDs
from agent import app as agent_app # Added for LangGraph agent
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage # Added
from typing import Sequence # Added for type hinting

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
PDFS_DIR = "pdfs" # Used for saving uploaded files
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
        st.session_state.chat_history = []  # Global chat history
    
    if 'agent_thread_id' not in st.session_state: # Added for agent conversation threading
        st.session_state.agent_thread_id = str(uuid.uuid4())

# Create necessary directories
def initialize_directories():
    """Create necessary directories if they don't exist."""
    if not os.path.exists(PDFS_DIR):
        os.makedirs(PDFS_DIR)
    if not os.path.exists(ASSETS_DIR): # Ensure assets dir is also checked/created if needed by app
        os.makedirs(ASSETS_DIR)


def get_user_stats():
    """Get statistics for the platform: total answers, total questions, and processed documents."""
    processed_docs_count = 0 # Simplified as PROCESSED_DOCS_FILE is removed; agent handles processing.
                             # This count might need a new source if "processed by agent" is to be tracked.
    
    total_questions = 0
    total_answers = 0
    if 'chat_history' in st.session_state:
        # Chat history is now a list of dicts
        for message in st.session_state.chat_history:
            if message.get("role") == "user":
                total_questions += 1
            elif message.get("role") == "assistant":
                total_answers += 1
    
    return total_answers, total_questions, processed_docs_count

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

def main():
    # Initialize
    init_session_state()
    initialize_directories()
    
    # Sidebar
    with st.sidebar:
        # Platform Statistics as a dropdown (expander)
        with st.expander("📊 Platform Statistics", expanded=True):
            try:
                total_answers, total_questions, processed_docs_count = get_user_stats()
                st.markdown(f"**Total Answers:** {total_answers}")
                st.markdown(f"**Total Questions:** {total_questions}")
                st.markdown(f"**Processed Documents:** {processed_docs_count}")
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
            st.info("The agent will process newly uploaded files based on its configuration.") # Simplified message

        # Show available documents - Simplified
        st.markdown("### 📚 Available Documents")
        st.caption("Uploaded documents are available to the agent in the 'pdfs' directory.")
    
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
                # Pass the entire chat history for context
                response_text = get_agent_response(st.session_state.chat_history, st.session_state.agent_thread_id)
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