"""Session state initialization and configuration loading"""

import streamlit as st
import os


def initialize_session_state():
    """Initialize session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = []
    if 'custom_metrics' not in st.session_state:
        st.session_state.custom_metrics = []
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'rag_documents' not in st.session_state:
        st.session_state.rag_documents = []
    if 'evaluation_data' not in st.session_state:
        st.session_state.evaluation_data = None
    if 'active_collection' not in st.session_state:
        st.session_state.active_collection = "rag_collection_eerrur_8415"
    if 'collection_files' not in st.session_state:
        st.session_state.collection_files = {}
    if 'selected_embedding_model' not in st.session_state:
        st.session_state.selected_embedding_model = None
    if 'selected_inference_model' not in st.session_state:
        st.session_state.selected_inference_model = None
    if 'selected_evaluation_model' not in st.session_state:
        st.session_state.selected_evaluation_model = None
    if 'selected_ocr_model' not in st.session_state:
        st.session_state.selected_ocr_model = None
    if 'extracted_texts' not in st.session_state:
        st.session_state.extracted_texts = {}


def load_config():
    """Load configuration from session state with defaults"""
    # Get models from selected models if available
    selected_embedding = st.session_state.get('selected_embedding_model', None)
    selected_inference = st.session_state.get('selected_inference_model', None)
    selected_evaluation = st.session_state.get('selected_evaluation_model', None)
    
    # Determine embedding config
    if selected_embedding:
        embedding_provider = "Custom Models"
        embedding_model = selected_embedding.get('name', '')
        embedding_api_key = selected_embedding.get('api_key', '')
    else:
        embedding_provider = st.session_state.get("embedding_provider", "OpenAI")
        embedding_model = st.session_state.get("embedding_model", "text-embedding-3-large")
        embedding_api_key = st.session_state.get("embedding_api_key", "")
    
    # Determine inference config
    if selected_inference:
        provider = "Custom Models"
        model_name = selected_inference.get('name', '')
        api_key = selected_inference.get('api_key', '')
    else:
        provider = st.session_state.get("provider", "OpenAI")
        model_name = st.session_state.get("model_name", "gpt-4o")
        api_key = st.session_state.get("api_key", "")
    
    # Determine evaluation config
    if selected_evaluation:
        eval_provider = "Custom Models"
        eval_model = selected_evaluation.get('name', '')
        eval_api_key = selected_evaluation.get('api_key', '')
    else:
        eval_provider = st.session_state.get("eval_provider", "Claude")
        eval_model = st.session_state.get("eval_model", st.session_state.get("claude_model", "claude-sonnet-4-20250514"))
        eval_api_key = st.session_state.get("eval_api_key", st.session_state.get("claude_api_key", ""))
    
    # Detect if running in Docker and adjust default Milvus host
    default_milvus_host = "localhost"
    if os.path.exists("/.dockerenv"):
        # Running inside Docker container, use service name
        default_milvus_host = "standalone"
    
    return {
        "milvus_host": st.session_state.get("milvus_host", default_milvus_host),
        "milvus_port": st.session_state.get("milvus_port", 19530),
        "milvus_user": st.session_state.get("milvus_user", ""),
        "milvus_password": st.session_state.get("milvus_password", ""),
        "provider": provider,
        "model_name": model_name,
        "api_key": api_key,
        "claude_api_key": st.session_state.get("claude_api_key", ""),  # Legacy support
        "eval_provider": eval_provider,
        "eval_api_key": eval_api_key,
        "eval_model": eval_model,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "embedding_api_key": embedding_api_key,
        "ollama_base_url": st.session_state.get("ollama_base_url", "http://localhost:11434"),
        "deepinfra_base_url": st.session_state.get("deepinfra_base_url", "https://api.deepinfra.com/v1/openai"),
        "embedding_openai_base_url": st.session_state.get("embedding_openai_base_url", st.session_state.get("deepinfra_base_url", "https://api.deepinfra.com/v1/openai")),
        "claude_model": st.session_state.get("claude_model", "claude-sonnet-4-20250514"),  # Legacy support
        "num_chunks": st.session_state.get("num_chunks", 5),
        "chunk_size": st.session_state.get("chunk_size", 1000),
        "chunk_overlap": st.session_state.get("chunk_overlap", 50),
        "temperature": st.session_state.get("temperature", 0.1)
    }
