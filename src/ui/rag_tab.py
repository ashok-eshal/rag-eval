"""RAG Documents Tab UI"""

import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from src.config.session import load_config
from src.config.constants import DEFAULT_COLLECTION
from src.utils.helpers import generate_random_collection_name
from src.utils.file_processing import process_uploaded_file
from src.services.embedding_service import create_embeddings
from src.services.vectorstore_service import setup_milvus_vectorstore, get_existing_milvus_vectorstore, get_collection_files


def render_rag_tab():
    """Render the RAG Documents tab"""
    st.header("RAG Knowledge Base Management")
    
    # Default Collection Section
    st.subheader("📁 Default Collection")
    default_col1, default_col2 = st.columns([2, 1])
    
    with default_col1:
        st.info(f"**Collection Name:** `{DEFAULT_COLLECTION}`")
        
        if (st.session_state.active_collection == DEFAULT_COLLECTION and 
            DEFAULT_COLLECTION in st.session_state.collection_files):
            files_list = st.session_state.collection_files[DEFAULT_COLLECTION]
            if files_list:
                st.markdown("**Files in this collection:**")
                for file in files_list:
                    st.markdown(f"• {file}")
            else:
                st.markdown("*Connect to see files in collection*")
        else:
            st.markdown("*Click 'Use Default Collection' to see files*")
    
    with default_col2:
        if st.button("✅ Use Default Collection", type="primary"):
            config = load_config()
            
            if config["embedding_provider"] == "OpenAI" and not config["embedding_api_key"]:
                st.error("Please provide OpenAI API key for embeddings in the Configuration tab")
            else:
                with st.spinner(f"Connecting to default collection '{DEFAULT_COLLECTION}'..."):
                    embeddings = create_embeddings(
                        provider=config["embedding_provider"],
                        model_name=config["embedding_model"],
                        api_key=config.get("embedding_api_key"),
                        ollama_base_url=config.get("ollama_base_url", "http://localhost:11434")
                    )
                    
                    if not embeddings:
                        st.error("Failed to create embeddings. Please check your configuration.")
                    else:
                        connection_args = {
                            "host": config["milvus_host"],
                            "port": config["milvus_port"]
                        }
                        if config["milvus_user"]:
                            connection_args["user"] = config["milvus_user"]
                        if config["milvus_password"]:
                            connection_args["password"] = config["milvus_password"]
                        
                        vectorstore = get_existing_milvus_vectorstore(
                            embeddings, 
                            connection_args,
                            DEFAULT_COLLECTION
                        )
                        
                        if vectorstore:
                            collection_files = get_collection_files(vectorstore, DEFAULT_COLLECTION)
                            
                            st.session_state.vectorstore = vectorstore
                            st.session_state.active_collection = DEFAULT_COLLECTION
                            st.session_state.collection_files[DEFAULT_COLLECTION] = collection_files
                            st.success(f"✅ Connected to default collection '{DEFAULT_COLLECTION}'!")
                            
                            if collection_files:
                                st.info(f"Found {len(collection_files)} files in the collection")
                            st.rerun()
                        else:
                            st.error("Failed to connect to default collection. Please check your configuration.")
    
    st.markdown("---")
    
    # Create New Collection Section
    st.subheader("📝 Create New Collection")
    
    if st.button("🆕 I want to upload files in a new collection"):
        new_collection_name = generate_random_collection_name()
        st.session_state.new_collection_name = new_collection_name
        st.session_state.show_upload = True
    
    if st.session_state.get('show_upload', False):
        st.info(f"**New Collection Name:** `{st.session_state.new_collection_name}`")
        
        uploaded_files = st.file_uploader(
            "Choose documents for new RAG knowledge base",
            type=['pdf', 'txt', 'docx', 'md'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} files selected")
            
            for file in uploaded_files:
                st.write(f"• {file.name} ({file.size / 1024:.2f} KB)")
            
            # OCR option
            use_ocr = False
            if st.session_state.get('selected_ocr_model'):
                use_ocr = st.checkbox(
                    "Use OCR for scanned PDFs",
                    value=False,
                    help="Enable OCR if any uploaded PDF files are scanned documents"
                )
                if use_ocr:
                    st.info("⚠️ OCR will be used for PDFs that appear to be scanned (no extractable text)")
            
            if st.button("📤 Process and Upload to New Collection"):
                config = load_config()
                
                if config["embedding_provider"] == "OpenAI" and not config["embedding_api_key"]:
                    st.error("Please provide OpenAI API key for embeddings in the Configuration tab")
                else:
                    with st.spinner("Processing documents..."):
                        all_documents = []
                        file_names = []
                        
                        for uploaded_file in uploaded_files:
                            text = process_uploaded_file(uploaded_file, use_ocr=use_ocr)
                            
                            if text:
                                doc = Document(
                                    page_content=text,
                                    metadata={"source": uploaded_file.name}
                                )
                                all_documents.append(doc)
                                file_names.append(uploaded_file.name)
                                
                                # Store extracted text in session state for persistence
                                st.session_state.extracted_texts[uploaded_file.name] = text
                                
                                st.info(f"Processed: {uploaded_file.name}")
                        
                        if all_documents:
                            text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=config["chunk_size"],
                                chunk_overlap=config["chunk_overlap"]
                            )
                            split_docs = text_splitter.split_documents(all_documents)
                            
                            embeddings = create_embeddings(
                                provider=config["embedding_provider"],
                                model_name=config["embedding_model"],
                                api_key=config.get("embedding_api_key"),
                                ollama_base_url=config.get("ollama_base_url", "http://localhost:11434")
                            )
                            
                            if not embeddings:
                                st.error("Failed to create embeddings. Please check your configuration.")
                            else:
                                connection_args = {
                                    "host": config["milvus_host"],
                                    "port": config["milvus_port"]
                                }
                                if config["milvus_user"]:
                                    connection_args["user"] = config["milvus_user"]
                                if config["milvus_password"]:
                                    connection_args["password"] = config["milvus_password"]
                                
                                vectorstore = setup_milvus_vectorstore(
                                    split_docs, 
                                    embeddings, 
                                    connection_args,
                                    st.session_state.new_collection_name
                                )
                                
                                if vectorstore:
                                    st.session_state.vectorstore = vectorstore
                                    st.session_state.active_collection = st.session_state.new_collection_name
                                    st.session_state.collection_files[st.session_state.new_collection_name] = file_names
                                    st.session_state.show_upload = False
                                    st.success(f"✅ Created new collection '{st.session_state.new_collection_name}' with {len(split_docs)} chunks from {len(all_documents)} documents!")
                                    st.rerun()
                                else:
                                    st.error("Failed to setup Milvus vectorstore")
                        else:
                            st.error("No text extracted from uploaded files")
    
    # Show extracted texts with persistence
    if st.session_state.get('extracted_texts') and st.session_state.extracted_texts:
        st.markdown("---")
        col_header1, col_header2 = st.columns([3, 1])
        with col_header1:
            st.subheader("📄 Extracted Text Previews")
        with col_header2:
            if st.button("🗑️ Clear All Previews", type="secondary"):
                st.session_state.extracted_texts = {}
                st.rerun()
        
        for filename, text in st.session_state.extracted_texts.items():
            with st.expander(f"📄 View extracted text: {filename}", expanded=False):
                st.text_area(
                    "Extracted Text",
                    text,
                    height=200,
                    disabled=True,
                    label_visibility="collapsed",
                    key=f"extracted_text_{filename}"
                )
                st.caption(f"Character count: {len(text)}")
    
    if st.session_state.active_collection:
        st.markdown("---")
        st.subheader("🔗 Currently Connected")
        st.success(f"Active Collection: `{st.session_state.active_collection}`")
        if st.session_state.active_collection in st.session_state.collection_files:
            st.markdown("**Files in active collection:**")
            for file in st.session_state.collection_files[st.session_state.active_collection]:
                st.markdown(f"• {file}")
