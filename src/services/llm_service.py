"""LLM service for initializing and managing different LLM providers"""

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_together import Together


def get_llm_provider(provider: str, model_name: str, api_key: str, temperature: float = 0.1):
    """Initialize LLM based on provider"""
    # Check if using custom inference model
    if provider == "Custom Models":
        selected_model = st.session_state.get('selected_inference_model', None)
        if not selected_model:
            raise ValueError("No custom inference model selected")
        
        model_name = selected_model.get('name', model_name)
        base_url = selected_model.get('base_url', 'https://api.deepinfra.com/v1/openai')
        custom_temperature = selected_model.get('temperature', temperature)
        fixed_temperature = selected_model.get('fixed_temperature', False)
        
        # Use API key from selected model if available, otherwise use provided key
        custom_api_key = selected_model.get('api_key', api_key)
        
        # Use fixed temperature if model requires it, otherwise use configured temperature
        final_temperature = 1.0 if fixed_temperature else custom_temperature
        
        return ChatOpenAI(
            model_name=model_name,
            openai_api_key=custom_api_key,
            base_url=base_url,
            temperature=final_temperature
        )
    elif provider == "OpenAI":
        # GPT-5 and O1 models don't support custom temperature (must use default)
        models_with_fixed_temperature = ['gpt-5', 'o1-preview', 'o1-mini', 'o1']
        is_fixed_temp_model = any(model in model_name.lower() for model in models_with_fixed_temperature)
        
        if is_fixed_temp_model:
            # GPT-5/O1 models: must use default temperature (1.0) - set explicitly to avoid ChatOpenAI default of 0.7
            return ChatOpenAI(
                model_name=model_name,
                openai_api_key=api_key,
                temperature=1.0  # GPT-5 requires default temperature of 1.0
            )
        else:
            return ChatOpenAI(
                model_name=model_name,
                openai_api_key=api_key,
                temperature=temperature
            )
    elif provider == "Groq":
        return ChatGroq(
            model_name=model_name,
            groq_api_key=api_key,
            temperature=temperature
        )
    elif provider == "Together AI":
        return Together(
            model=model_name,
            together_api_key=api_key,
            temperature=temperature
        )
    elif provider == "DeepInfra":
        # Uses OpenAI-compatible API with custom base_url
        base_url = st.session_state.get("deepinfra_base_url", "https://api.deepinfra.com/v1/openai")
        # For safety, keep temperature behavior similar to OpenAI handling
        models_with_fixed_temperature = ['gpt-5', 'o1-preview', 'o1-mini', 'o1']
        is_fixed_temp_model = any(model in model_name.lower() for model in models_with_fixed_temperature)
        if is_fixed_temp_model:
            return ChatOpenAI(
                model_name=model_name,
                openai_api_key=api_key,
                base_url=base_url,
                temperature=1.0
            )
        else:
            return ChatOpenAI(
                model_name=model_name,
                openai_api_key=api_key,
                base_url=base_url,
                temperature=temperature
            )
    else:
        raise ValueError(f"Unsupported provider: {provider}")
