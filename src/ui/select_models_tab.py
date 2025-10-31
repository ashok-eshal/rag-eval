"""Select Models Tab UI for DeepInfra Model Management"""

import streamlit as st
from src.config.constants import BASE_URLS, PROVIDERS


def render_select_models_tab():
    """Render the Select Models tab for managing DeepInfra models"""
    st.header("🔌 Select Models")
    st.markdown("Configure and manage embedding and inference models from DeepInfra or other providers")
    
    # Initialize session state for selected models (only one of each type)
    if 'selected_embedding_model' not in st.session_state:
        st.session_state.selected_embedding_model = None
    if 'selected_inference_model' not in st.session_state:
        st.session_state.selected_inference_model = None
    if 'selected_evaluation_model' not in st.session_state:
        st.session_state.selected_evaluation_model = None
    if 'selected_ocr_model' not in st.session_state:
        st.session_state.selected_ocr_model = None
    
    # Tab for Embedding Models, Inference Models, Evaluation Models, and OCR Models
    model_tab1, model_tab2, model_tab3, model_tab4 = st.tabs([
        "📊 Embedding Models",
        "🤖 Inference Models",
        "🔍 Evaluation Models",
        "👁️ OCR Models"
    ])
    
    with model_tab1:
        st.subheader("Embedding Model Configuration")
        st.info("""
        Embedding models convert text into numerical vectors that can be searched and compared.
        Configure your embedding model from DeepInfra or other OpenAI-compatible providers.
        Note: Dimension selection is automatic for DeepInfra models.
        """)
        
        # Embedding Model Configuration Form
        with st.form("embedding_model_form"):
            model_name = st.text_input(
                "Model Name *",
                value=st.session_state.selected_embedding_model.get('name', '') if st.session_state.selected_embedding_model else '',
                placeholder="e.g., BAAI/bge-base-en-v1.5",
                help="Full model name as provided by the provider (DeepInfra will auto-select dimensions)"
            )
            
            api_key = st.text_input(
                "API Key *",
                value=st.session_state.selected_embedding_model.get('api_key', '') if st.session_state.selected_embedding_model else '',
                type="password",
                placeholder="Enter API key for authentication",
                help="API key required for connecting to the embedding model"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Since we only have DeepInfra, just set the base URL directly
                base_url = BASE_URLS["DeepInfra"]
                st.info(f"Base URL: `{base_url}`")
            
            with col2:
                current_provider_value = st.session_state.selected_embedding_model.get('provider', 'DeepInfra') if st.session_state.selected_embedding_model else 'DeepInfra'
                provider_idx = PROVIDERS.index(current_provider_value) if current_provider_value in PROVIDERS else 0
                
                provider_name = st.selectbox(
                    "Provider",
                    PROVIDERS,
                    index=provider_idx,
                    help="Select the model provider"
                )
            
            submitted = st.form_submit_button("💾 Save", type="primary")
            
            if submitted:
                if not model_name or not base_url or not api_key:
                    st.error("❌ Model Name, Base URL, and API Key are required!")
                else:
                    st.session_state.selected_embedding_model = {
                        "name": model_name,
                        "api_key": api_key,
                        "dimensions": None,  # DeepInfra will auto-select dimensions
                        "base_url": base_url,
                        "provider": provider_name,
                        "requires_api_key": True
                    }
                    st.success(f"✅ Embedding model saved: {model_name}")
        
        # Display current selected model
        if st.session_state.selected_embedding_model:
            st.markdown("### Current Selected Embedding Model")
            model_info = st.session_state.selected_embedding_model
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Provider:** {model_info.get('provider', 'Unknown')}")
                st.write(f"**Base URL:** {model_info.get('base_url', 'N/A')}")
            with col2:
                st.write(f"**Dimensions:** Auto (DeepInfra)")
                st.write(f"**Model:** {model_info.get('name', 'N/A')}")
    
    with model_tab2:
        st.subheader("Inference Model Configuration")
        st.info("""
        Inference models generate text responses. Configure your inference model from DeepInfra or other providers.
        These models will be used for generating responses in the RAG system.
        """)
        
        # Inference Model Configuration Form
        with st.form("inference_model_form"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                model_name = st.text_input(
                    "Model Name *",
                    value=st.session_state.selected_inference_model.get('name', '') if st.session_state.selected_inference_model else '',
                    placeholder="e.g., meta-llama/Meta-Llama-3.1-70B-Instruct",
                    help="Full model name as provided by the provider"
                )
            
            with col2:
                fixed_temperature = st.checkbox(
                    "Fixed Temperature",
                    value=st.session_state.selected_inference_model.get('fixed_temperature', False) if st.session_state.selected_inference_model else False,
                    help="Some models (like GPT-5, O1) require fixed temperature"
                )
            
            api_key = st.text_input(
                "API Key *",
                value=st.session_state.selected_inference_model.get('api_key', '') if st.session_state.selected_inference_model else '',
                type="password",
                placeholder="Enter API key for authentication",
                help="API key required for connecting to the inference model"
            )
            
            col3, col4 = st.columns(2)
            
            with col3:
                # Since we only have DeepInfra, just set the base URL directly
                base_url = BASE_URLS["DeepInfra"]
                st.info(f"Base URL: `{base_url}`")
            
            with col4:
                current_provider_value = st.session_state.selected_inference_model.get('provider', 'DeepInfra') if st.session_state.selected_inference_model else 'DeepInfra'
                provider_idx = PROVIDERS.index(current_provider_value) if current_provider_value in PROVIDERS else 0
                
                provider_name = st.selectbox(
                    "Provider",
                    PROVIDERS,
                    index=provider_idx,
                    help="Select the model provider"
                )
            
            col5 = st.columns(1)[0]
            
            with col5:
                temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=2.0,
                    value=st.session_state.selected_inference_model.get('temperature', 0.1) if st.session_state.selected_inference_model else 0.1,
                    step=0.1,
                    help="Control randomness in outputs (0.0 = deterministic, 2.0 = very creative)"
                )
            
            submitted = st.form_submit_button("💾 Save", type="primary")
            
            if submitted:
                if not model_name or not base_url or not api_key:
                    st.error("❌ Model Name, Base URL, and API Key are required!")
                else:
                    st.session_state.selected_inference_model = {
                        "name": model_name,
                        "api_key": api_key,
                        "base_url": base_url,
                        "provider": provider_name,
                        "fixed_temperature": fixed_temperature,
                        "temperature": temperature
                    }
                    st.success(f"✅ Inference model saved: {model_name}")
        
        # Display current selected model
        if st.session_state.selected_inference_model:
            st.markdown("### Current Selected Inference Model")
            model_info = st.session_state.selected_inference_model
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Provider:** {model_info.get('provider', 'Unknown')}")
                st.write(f"**Base URL:** {model_info.get('base_url', 'N/A')}")
            with col2:
                st.write(f"**Temperature:** {model_info.get('temperature', 0.1)}")
                fixed_temp = "✅ Yes" if model_info.get('fixed_temperature', False) else "❌ No"
                st.write(f"**Fixed Temperature:** {fixed_temp}")
                st.write(f"**Model:** {model_info.get('name', 'N/A')}")
    
    with model_tab3:
        st.subheader("Evaluation Model Configuration")
        st.info("""
        Evaluation models are used for assessing LLM responses. Configure your evaluation model from providers like Claude or OpenAI.
        These models will be used for evaluating responses in the evaluation process.
        """)
        
        # Evaluation Model Configuration Form
        with st.form("evaluation_model_form"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                model_name = st.text_input(
                    "Model Name *",
                    value=st.session_state.selected_evaluation_model.get('name', '') if st.session_state.selected_evaluation_model else '',
                    placeholder="e.g., claude-sonnet-4-20250514",
                    help="Full model name as provided by the provider"
                )
            
            with col2:
                fixed_temperature = st.checkbox(
                    "Fixed Temperature",
                    value=st.session_state.selected_evaluation_model.get('fixed_temperature', False) if st.session_state.selected_evaluation_model else False,
                    help="Some models (like GPT-5, O1) require fixed temperature"
                )
            
            api_key = st.text_input(
                "API Key *",
                value=st.session_state.selected_evaluation_model.get('api_key', '') if st.session_state.selected_evaluation_model else '',
                type="password",
                placeholder="Enter API key for authentication",
                help="API key required for connecting to the evaluation model"
            )
            
            col3, col4 = st.columns(2)
            
            with col3:
                # Since we only have DeepInfra, just set the base URL directly
                base_url = BASE_URLS["DeepInfra"]
                st.info(f"Base URL: `{base_url}`")
            
            with col4:
                current_provider_value = st.session_state.selected_evaluation_model.get('provider', 'DeepInfra') if st.session_state.selected_evaluation_model else 'DeepInfra'
                provider_idx = PROVIDERS.index(current_provider_value) if current_provider_value in PROVIDERS else 0
                
                provider_name = st.selectbox(
                    "Provider",
                    PROVIDERS,
                    index=provider_idx,
                    help="Select the model provider"
                )
            
            col5 = st.columns(1)[0]
            
            with col5:
                temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=2.0,
                    value=st.session_state.selected_evaluation_model.get('temperature', 0.1) if st.session_state.selected_evaluation_model else 0.1,
                    step=0.1,
                    help="Control randomness in outputs (0.0 = deterministic, 2.0 = very creative)"
                )
            
            submitted = st.form_submit_button("💾 Save", type="primary")
            
            if submitted:
                if not model_name or not base_url or not api_key:
                    st.error("❌ Model Name, Base URL, and API Key are required!")
                else:
                    st.session_state.selected_evaluation_model = {
                        "name": model_name,
                        "api_key": api_key,
                        "base_url": base_url,
                        "provider": provider_name,
                        "fixed_temperature": fixed_temperature,
                        "temperature": temperature
                    }
                    st.success(f"✅ Evaluation model saved: {model_name}")
        
        # Display current selected model
        if st.session_state.selected_evaluation_model:
            st.markdown("### Current Selected Evaluation Model")
            model_info = st.session_state.selected_evaluation_model
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Provider:** {model_info.get('provider', 'Unknown')}")
                st.write(f"**Base URL:** {model_info.get('base_url', 'N/A')}")
            with col2:
                st.write(f"**Temperature:** {model_info.get('temperature', 0.1)}")
                fixed_temp = "✅ Yes" if model_info.get('fixed_temperature', False) else "❌ No"
                st.write(f"**Fixed Temperature:** {fixed_temp}")
                st.write(f"**Model:** {model_info.get('name', 'N/A')}")
    
    with model_tab4:
        st.subheader("OCR Model Configuration")
        st.info("""
        OCR (Optical Character Recognition) models extract text from scanned PDFs and images.
        Configure your OCR model to enable text extraction from scanned documents.
        """)
        
        # OCR Model Configuration Form
        with st.form("ocr_model_form"):
            model_name = st.text_input(
                "Model Name *",
                value=st.session_state.selected_ocr_model.get('name', '') if st.session_state.selected_ocr_model else '',
                placeholder="e.g., qwen/Qwen2-VL-7B-Instruct or microsoft/unilm-dit-layoutlm",
                help="Full model name as provided by the provider (vision-based models recommended)"
            )
            
            api_key = st.text_input(
                "API Key *",
                value=st.session_state.selected_ocr_model.get('api_key', '') if st.session_state.selected_ocr_model else '',
                type="password",
                placeholder="Enter API key for authentication",
                help="API key required for connecting to the OCR model"
            )
            
            col3, col4 = st.columns(2)
            
            with col3:
                # Since we only have DeepInfra, just set the base URL directly
                base_url = BASE_URLS["DeepInfra"]
                st.info(f"Base URL: `{base_url}`")
            
            with col4:
                current_provider_value = st.session_state.selected_ocr_model.get('provider', 'DeepInfra') if st.session_state.selected_ocr_model else 'DeepInfra'
                provider_idx = PROVIDERS.index(current_provider_value) if current_provider_value in PROVIDERS else 0
                
                provider_name = st.selectbox(
                    "Provider",
                    PROVIDERS,
                    index=provider_idx,
                    help="Select the model provider"
                )
            
            submitted = st.form_submit_button("💾 Save", type="primary")
            
            if submitted:
                if not model_name or not base_url or not api_key:
                    st.error("❌ Model Name, Base URL, and API Key are required!")
                else:
                    st.session_state.selected_ocr_model = {
                        "name": model_name,
                        "api_key": api_key,
                        "base_url": base_url,
                        "provider": provider_name
                    }
                    st.success(f"✅ OCR model saved: {model_name}")
        
        # Display current selected model
        if st.session_state.selected_ocr_model:
            st.markdown("### Current Selected OCR Model")
            model_info = st.session_state.selected_ocr_model
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Provider:** {model_info.get('provider', 'Unknown')}")
                st.write(f"**Base URL:** {model_info.get('base_url', 'N/A')}")
            with col2:
                st.write(f"**Model:** {model_info.get('name', 'N/A')}")
    
    # Summary
    st.markdown("---")
    st.markdown("### 📊 Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        embedding_status = "✅ Configured" if st.session_state.selected_embedding_model else "❌ Not Configured"
        st.metric("Embedding Model", embedding_status)
    with col2:
        inference_status = "✅ Configured" if st.session_state.selected_inference_model else "❌ Not Configured"
        st.metric("Inference Model", inference_status)
    with col3:
        evaluation_status = "✅ Configured" if st.session_state.selected_evaluation_model else "❌ Not Configured"
        st.metric("Evaluation Model", evaluation_status)
    with col4:
        ocr_status = "✅ Configured" if st.session_state.selected_ocr_model else "❌ Not Configured"
        st.metric("OCR Model", ocr_status)
