"""Configuration Tab UI"""

import streamlit as st
from src.config.session import load_config
from src.services.vectorstore_service import test_milvus_connection


def render_config_tab():
    """Render the Configuration tab"""
    st.header("System Configuration")
    
    config = load_config()
    
    # Milvus Configuration
    st.subheader("Milvus Configuration")
    col1, col2 = st.columns(2)
    with col1:
        milvus_host = st.text_input("Milvus Host", value=config["milvus_host"], key="widget_milvus_host")
        milvus_user = st.text_input("Milvus Username", value=config["milvus_user"], key="widget_milvus_user")
    with col2:
        milvus_port = st.number_input("Milvus Port", value=config["milvus_port"], step=1, key="widget_milvus_port")
        milvus_password = st.text_input("Milvus Password", value=config["milvus_password"], type="password", key="widget_milvus_password")
    
    # Test Milvus Connection
    if st.button("🔌 Test Milvus Connection", type="secondary"):
        connection_args = {
            "host": milvus_host,
            "port": int(milvus_port)
        }
        if milvus_user:
            connection_args["user"] = milvus_user
        if milvus_password:
            connection_args["password"] = milvus_password
        
        with st.spinner("Testing connection to Milvus..."):
            is_connected, msg = test_milvus_connection(connection_args)
            if is_connected:
                st.success(f"✅ {msg}")
            else:
                st.error(f"❌ {msg}")
                st.info("💡 **Quick Start:** Run Milvus with Docker:\n```bash\ndocker run -d --name milvus-standalone -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest\n```\n\nOr check if your Milvus container is running:\n```bash\ndocker ps | grep milvus\n```")
    
    # Model configuration note
    st.info("💡 **Models:** Configure all models (Embedding, Inference, and Evaluation) in the 'Select Models' tab.")
    
    # RAG Configuration
    st.subheader("RAG Configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        num_chunks = st.slider("Number of Chunks to Retrieve", min_value=1, max_value=10, 
                              value=config["num_chunks"])
    with col2:
        chunk_size = st.slider("Chunk Size", min_value=100, max_value=2000, 
                              value=config["chunk_size"], step=100)
    with col3:
        chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=200, 
                                 value=config["chunk_overlap"], step=10)
    
    # Save configuration
    if st.button("💾 Save Configuration"):
        # Extract model info from selected models if available
        selected_inference = st.session_state.get('selected_inference_model', None)
        selected_evaluation = st.session_state.get('selected_evaluation_model', None)
        selected_embedding = st.session_state.get('selected_embedding_model', None)
        
        update_dict = {
            "milvus_host": milvus_host,
            "milvus_port": milvus_port,
            "milvus_user": milvus_user,
            "milvus_password": milvus_password,
            # Set models to "Custom Models" if selected, otherwise keep defaults
            "provider": "Custom Models" if selected_inference else config.get("provider", "OpenAI"),
            "model_name": selected_inference.get('name', '') if selected_inference else config.get("model_name", "gpt-4o"),
            "api_key": "",  # API keys are stored in selected models
            "eval_provider": "Custom Models" if selected_evaluation else config.get("eval_provider", "Claude"),
            "eval_api_key": "",  # API keys are stored in selected models
            "eval_model": selected_evaluation.get('name', '') if selected_evaluation else config.get("eval_model", "claude-sonnet-4-20250514"),
            "embedding_provider": "Custom Models" if selected_embedding else config.get("embedding_provider", "OpenAI"),
            "embedding_api_key": "",  # API keys are stored in selected models
            "embedding_model": selected_embedding.get('name', '') if selected_embedding else config.get("embedding_model", "text-embedding-3-large"),
            "ollama_base_url": "http://localhost:11434",  # Default
            "num_chunks": num_chunks,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap
        }
        st.session_state.update(update_dict)
        st.success("✅ Configuration saved!")
