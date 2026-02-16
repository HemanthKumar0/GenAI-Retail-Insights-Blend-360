"""
Retail Insights Assistant - Streamlit User Interface

This is the main Streamlit application that provides a web-based chat interface
for interacting with the Retail Insights Assistant.

**Validates: Requirements 14.1, 14.2, 14.3, 14.4, 14.5, 14.6**
"""

import streamlit as st
from pathlib import Path
import pandas as pd
from datetime import datetime
import traceback

from src.utils.app_utils import create_orchestrator
from src.core.models import Message
from src.core.config import Config

# Page configuration
st.set_page_config(
    page_title="Retail Insights Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """
    Initialize Streamlit session state variables.
    
    **Validates: Requirement 11.4**
    """
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = create_orchestrator()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'mode' not in st.session_state:
        st.session_state.mode = "qa"
    
    if 'loaded_datasets' not in st.session_state:
        st.session_state.loaded_datasets = []
    
    if 'processing' not in st.session_state:
        st.session_state.processing = False


def display_header():
    """Display the application header."""
    st.markdown('<div class="main-header">📊 Retail Insights Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">AI-powered analysis of retail sales data through natural language</div>',
        unsafe_allow_html=True
    )


def display_sidebar():
    """
    Display the sidebar with file upload and mode selection.
    
    **Validates: Requirements 14.3, 14.4**
    """
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Mode selection
        st.subheader("Operating Mode")
        mode = st.radio(
            "Select mode:",
            options=["qa", "summarization"],
            format_func=lambda x: "💬 Q&A Mode" if x == "qa" else "📝 Summarization Mode",
            index=0 if st.session_state.mode == "qa" else 1,
            help="Q&A Mode: Ask specific questions about your data\nSummarization Mode: Get automatic insights and summaries"
        )
        
        if mode != st.session_state.mode:
            st.session_state.mode = mode
            st.rerun()
        
        st.divider()
        
        # File upload
        st.subheader("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload your sales data",
            type=["csv", "xlsx", "json"],
            help="Supported formats: CSV, Excel (.xlsx), JSON"
        )
        
        if uploaded_file is not None:
            with st.spinner("Loading dataset..."):
                try:
                    # Save uploaded file temporarily
                    temp_path = Path("temp_uploads")
                    temp_path.mkdir(exist_ok=True)
                    file_path = temp_path / uploaded_file.name
                    
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Load into data store
                    orchestrator = st.session_state.orchestrator
                    data_store = orchestrator.extraction_agent.data_store
                    
                    if uploaded_file.name.endswith('.csv'):
                        table_name = data_store.load_csv(str(file_path))
                    elif uploaded_file.name.endswith('.xlsx'):
                        table_names = data_store.load_excel(str(file_path))
                        table_name = table_names[0] if table_names else None
                    elif uploaded_file.name.endswith('.json'):
                        table_name = data_store.load_json(str(file_path))
                    
                    if table_name and table_name not in st.session_state.loaded_datasets:
                        st.session_state.loaded_datasets.append(table_name)
                        st.success(f"✅ Loaded: {table_name}")
                        
                        # Get dataset info
                        schema = data_store.get_table_schema(table_name)
                        st.info(f"📊 {schema['row_count']} rows, {len(schema['columns'])} columns")
                    
                except Exception as e:
                    st.error(f"❌ Error loading file: {str(e)}")
        
        # Display loaded datasets
        if st.session_state.loaded_datasets:
            st.divider()
            st.subheader("📊 Loaded Datasets")
            for dataset in st.session_state.loaded_datasets:
                st.text(f"• {dataset}")
        
        # API Configuration status
        st.divider()
        st.subheader("🔑 API Status")
        if Config.OPENAI_API_KEY:
            st.success("✅ OpenAI API configured")
        elif Config.GOOGLE_API_KEY:
            st.success("✅ Google API configured")
        else:
            st.warning("⚠️ No API key configured")
            st.caption("Set OPENAI_API_KEY or GOOGLE_API_KEY in .env file")


def display_chat_history():
    """
    Display the conversation history.
    
    **Validates: Requirement 14.2**
    """
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        
        if role == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(content)
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(content)
                
                # Display data table if available
                if "data" in msg and msg["data"] is not None:
                    st.dataframe(msg["data"], use_container_width=True)
                
                # Display metadata
                if "metadata" in msg:
                    metadata = msg["metadata"]
                    cols = st.columns(3)
                    with cols[0]:
                        st.caption(f"⏱️ {metadata.get('execution_time', 0):.2f}s")
                    with cols[1]:
                        st.caption(f"🎯 {metadata.get('tokens_used', 0)} tokens")
                    with cols[2]:
                        if metadata.get('cached', False):
                            st.caption("💾 Cached")


def process_user_input(user_input: str):
    """
    Process user input and generate response.
    
    **Validates: Requirements 14.5, 14.6**
    """
    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now()
    })
    
    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)
    
    # Process query with loading indicator
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            try:
                orchestrator = st.session_state.orchestrator
                mode = st.session_state.mode
                
                # Check if data is loaded
                if not st.session_state.loaded_datasets and mode == "qa":
                    response_text = (
                        "⚠️ No dataset loaded yet. Please upload a CSV, Excel, or JSON file "
                        "using the sidebar to get started."
                    )
                    st.warning(response_text)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text,
                        "timestamp": datetime.now()
                    })
                    return
                
                # Process query
                response = orchestrator.process_query(user_input, mode=mode)
                
                # Display response
                st.markdown(response.answer)
                
                # Display data table if available
                if response.data is not None and not response.data.empty:
                    st.dataframe(response.data, use_container_width=True)
                
                # Display metadata
                metadata = response.metadata
                cols = st.columns(3)
                with cols[0]:
                    st.caption(f"⏱️ {metadata.get('execution_time', 0):.2f}s")
                with cols[1]:
                    st.caption(f"🎯 {metadata.get('tokens_used', 0)} tokens")
                with cols[2]:
                    if metadata.get('cached', False):
                        st.caption("💾 Cached")
                
                # Add assistant message to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.answer,
                    "data": response.data,
                    "metadata": metadata,
                    "timestamp": datetime.now()
                })
                
            except Exception as e:
                error_msg = f"❌ An error occurred: {str(e)}"
                st.error(error_msg)
                
                # Log full error
                st.expander("Error Details").code(traceback.format_exc())
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "timestamp": datetime.now()
                })


def display_example_queries():
    """Display example queries based on current mode."""
    mode = st.session_state.mode
    
    st.subheader("💡 Example Queries")
    
    if mode == "qa":
        examples = [
            "What are the total sales?",
            "Show me the top 10 products by revenue",
            "What were the sales last month?",
            "Which category has the highest sales?",
            "Compare sales between Q1 and Q2"
        ]
    else:
        examples = [
            "Provide a summary of the sales data",
            "Analyze the key trends in the dataset",
            "What are the main insights from this data?"
        ]
    
    cols = st.columns(len(examples))
    for i, example in enumerate(examples):
        with cols[i]:
            if st.button(example, key=f"example_{i}", use_container_width=True):
                process_user_input(example)
                st.rerun()


def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Display header
    display_header()
    
    # Display sidebar
    display_sidebar()
    
    # Main content area
    if not st.session_state.loaded_datasets:
        # Welcome screen
        st.info("👋 Welcome! Please upload a dataset using the sidebar to get started.")
        
        st.markdown("""
        ### How to use:
        1. **Upload your data**: Use the sidebar to upload a CSV, Excel, or JSON file
        2. **Choose a mode**: Select between Q&A mode (ask specific questions) or Summarization mode (get automatic insights)
        3. **Start chatting**: Ask questions about your data in natural language
        
        ### Features:
        - 💬 Natural language queries
        - 📊 Automatic data analysis
        - 🔍 Smart insights and trends
        - 📈 Data visualization
        - 💾 Query caching for faster responses
        """)
    else:
        # Display example queries
        if not st.session_state.messages:
            display_example_queries()
        
        # Display chat history
        display_chat_history()
        
        # Chat input
        user_input = st.chat_input("Ask a question about your data...")
        
        if user_input:
            process_user_input(user_input)
            st.rerun()
    
    # Footer
    st.divider()
    st.caption("Retail Insights Assistant | Powered by GenAI")


if __name__ == "__main__":
    main()
