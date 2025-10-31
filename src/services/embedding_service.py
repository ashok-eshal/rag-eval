"""Embedding service for initializing and managing different embedding providers"""

import streamlit as st
from typing import List
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings, OllamaEmbeddings, SentenceTransformerEmbeddings

# OpenAI-compatible client for custom embeddings (e.g., DeepInfra)
try:
    from openai import OpenAI as OpenAIClient
    OPENAI_CLIENT_AVAILABLE = True
except Exception:
    OPENAI_CLIENT_AVAILABLE = False


def create_embeddings(provider: str, model_name: str, api_key: str = None, ollama_base_url: str = "http://localhost:11434", base_url_override: str = None):
    """Create embeddings based on provider and model selection"""
    try:
        if provider == "OpenAI":
            if not api_key:
                raise ValueError("OpenAI API key is required")
            return OpenAIEmbeddings(
                openai_api_key=api_key,
                model=model_name
            )
        elif provider == "OpenAI-Compatible" or provider == "Custom Models":
            # e.g., DeepInfra OpenAI-compatible endpoint or custom models
            if not api_key:
                raise ValueError("API key is required for OpenAI-Compatible embeddings")
            
            # Use override if provided, otherwise get from session state
            if base_url_override:
                base_url = base_url_override
            elif provider == "Custom Models":
                # Get base URL and API key from selected model info
                selected_model = st.session_state.get('selected_embedding_model', {})
                if selected_model:
                    base_url = selected_model.get('base_url', "https://api.deepinfra.com/v1/openai")
                    # Override API key with the one from selected model if available
                    if selected_model.get('api_key'):
                        api_key = selected_model.get('api_key')
                else:
                    base_url = st.session_state.get("embedding_openai_base_url", st.session_state.get("deepinfra_base_url", "https://api.deepinfra.com/v1/openai"))
            else:
                base_url = st.session_state.get("embedding_openai_base_url", st.session_state.get("deepinfra_base_url", "https://api.deepinfra.com/v1/openai"))
            
            if not OPENAI_CLIENT_AVAILABLE:
                raise ValueError("openai package not available. Please install openai>=1.0.0")

            class OpenAICompatibleEmbeddings:
                def __init__(self, api_key: str, base_url: str, model: str):
                    self.client = OpenAIClient(api_key=api_key, base_url=base_url)
                    self.model = model

                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    # Ensure inputs are strings
                    safe_texts = [t if isinstance(t, str) else str(t) for t in texts]
                    resp = self.client.embeddings.create(model=self.model, input=safe_texts, encoding_format="float")
                    return [d.embedding for d in resp.data]

                def embed_query(self, text: str) -> List[float]:
                    safe_text = text if isinstance(text, str) else str(text)
                    resp = self.client.embeddings.create(model=self.model, input=safe_text, encoding_format="float")
                    return resp.data[0].embedding

            return OpenAICompatibleEmbeddings(api_key=api_key, base_url=base_url, model=model_name)
        elif provider == "HuggingFace":
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu', 'trust_remote_code': True},
                encode_kwargs={'normalize_embeddings': True}
            )
        elif provider == "SentenceTransformers":
            return SentenceTransformerEmbeddings(
                model_name=model_name
            )
        elif provider == "Ollama":
            return OllamaEmbeddings(
                model=model_name,
                base_url=ollama_base_url
            )
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        return None
