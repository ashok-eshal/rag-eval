"""Vectorstore service for Milvus operations"""

import streamlit as st
from typing import List, Dict, Any, Tuple
from langchain.vectorstores import Milvus
from langchain.schema import Document

# Milvus direct connection for testing
try:
    from pymilvus import connections, utility
    PYMILVUS_AVAILABLE = True
except ImportError:
    PYMILVUS_AVAILABLE = False


def test_milvus_connection(connection_args: Dict[str, Any]) -> Tuple[bool, str]:
    """Test Milvus connection before attempting to use it"""
    if not PYMILVUS_AVAILABLE:
        return False, "pymilvus not available. Install it with: pip install pymilvus"
    
    try:
        # Try to connect and list collections as a test
        connections.connect(
            alias="default",
            host=connection_args.get("host", "localhost"),
            port=connection_args.get("port", 19530),
            user=connection_args.get("user", ""),
            password=connection_args.get("password", "")
        )
        # Test by listing collections
        utility.list_collections()
        connections.disconnect("default")
        return True, "Connection successful!"
    except Exception as e:
        error_msg = str(e)
        if "connection" in error_msg.lower() or "unavailable" in error_msg.lower():
            return False, f"Cannot connect to Milvus server. Please ensure:\n1. Milvus is running (check with: docker ps)\n2. Host and port are correct ({connection_args.get('host', 'localhost')}:{connection_args.get('port', 19530)})\n3. If using Docker, ensure ports are exposed correctly\n\nError: {error_msg}"
        elif "authentication" in error_msg.lower():
            return False, f"Authentication failed. Please check username/password.\n\nError: {error_msg}"
        else:
            return False, f"Connection test failed: {error_msg}"
    finally:
        # Ensure connection is closed
        try:
            connections.disconnect("default")
        except:
            pass


def setup_milvus_vectorstore(documents: List[Document], embeddings, connection_args: Dict[str, Any], collection_name: str):
    """Setup Milvus vector store"""
    # Test connection first
    is_connected, msg = test_milvus_connection(connection_args)
    if not is_connected:
        st.error(f"❌ Milvus Connection Test Failed\n\n{msg}")
        return None
    
    try:
        vectorstore = Milvus.from_documents(
            documents=documents,
            embedding=embeddings,
            connection_args=connection_args,
            collection_name=collection_name
        )
        return vectorstore
    except Exception as e:
        error_msg = str(e)
        if "connection" in error_msg.lower() or "unavailable" in error_msg.lower():
            st.error(f"❌ Error setting up Milvus: {error_msg}\n\n💡 Make sure Milvus is running:\n```bash\ndocker run -d --name milvus-standalone -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest\n```")
        else:
            st.error(f"❌ Error setting up Milvus: {error_msg}")
        return None


def get_existing_milvus_vectorstore(embeddings, connection_args: Dict[str, Any], collection_name: str):
    """Connect to existing Milvus collection"""
    # Test connection first
    is_connected, msg = test_milvus_connection(connection_args)
    if not is_connected:
        st.error(f"❌ Milvus Connection Test Failed\n\n{msg}")
        return None
    
    try:
        vectorstore = Milvus(
            embedding_function=embeddings,
            connection_args=connection_args,
            collection_name=collection_name
        )
        return vectorstore
    except Exception as e:
        error_msg = str(e)
        if "connection" in error_msg.lower() or "unavailable" in error_msg.lower():
            st.error(f"❌ Error connecting to existing Milvus collection: {error_msg}\n\n💡 Make sure Milvus is running:\n```bash\ndocker run -d --name milvus-standalone -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest\n```\n\nOr check if Milvus container is running:\n```bash\ndocker ps | grep milvus\n```")
        else:
            st.error(f"❌ Error connecting to existing Milvus collection: {error_msg}")
        return None


def get_collection_files(vectorstore, collection_name: str) -> List[str]:
    """Get unique source files from Milvus collection"""
    try:
        sample_docs = vectorstore.similarity_search("", k=100)
        sources = set()
        for doc in sample_docs:
            if doc.metadata and 'source' in doc.metadata:
                sources.add(doc.metadata['source'])
        return sorted(list(sources))
    except Exception as e:
        st.warning(f"Could not retrieve files from collection: {str(e)}")
        return []
