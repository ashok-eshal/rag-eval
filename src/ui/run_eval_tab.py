"""Run Evaluation Tab UI"""

import streamlit as st
import numpy as np
from datetime import datetime

from src.config.session import load_config
from src.config.constants import DEFAULT_METRICS, EMBEDDING_MODELS
from src.services.llm_service import get_llm_provider
from src.services.rag_service import create_rag_chain, generate_rag_response
from src.services.evaluation_service import evaluate_response


def render_run_eval_tab():
    """Render the Run Evaluation tab"""
    st.header("Run Evaluation")
    
    rag_ready = st.session_state.vectorstore is not None
    eval_ready = st.session_state.evaluation_data is not None
    config = load_config()
    
    if rag_ready and eval_ready:
        st.subheader("📊 Evaluation Configuration Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### RAG Configuration")
            
            embedding_info = ""
            if config.get("embedding_provider") and config.get("embedding_model"):
                provider_models = EMBEDDING_MODELS.get(config["embedding_provider"], {})
                model_info = provider_models.get(config["embedding_model"], {})
                embedding_info = f"""
            **Embedding Provider:** {config['embedding_provider']}  
            **Embedding Model:** {config['embedding_model']}  
            **Embedding Dimensions:** {model_info.get('dimensions', 'Unknown')}"""
            
            st.info(f"""
            **Active Collection:** `{st.session_state.active_collection}`  
            **Files in Collection:** {len(st.session_state.collection_files.get(st.session_state.active_collection, []))}  
            {embedding_info}
            **Chunks to Retrieve:** {config['num_chunks']}  
            **Chunk Size:** {config['chunk_size']}  
            **Chunk Overlap:** {config['chunk_overlap']}
            """)
            
            if st.session_state.active_collection in st.session_state.collection_files:
                with st.expander("View files in collection"):
                    for file in st.session_state.collection_files[st.session_state.active_collection]:
                        st.write(f"• {file}")
        
        with col2:
            st.markdown("### Model Configuration")
            st.info(f"""
            **LLM Provider:** {config['provider']}  
            **LLM Model:** {config['model_name']}  
            **Temperature:** {config['temperature']}  
            **Evaluation Provider:** {config.get('eval_provider', 'Claude')}  
            **Evaluation Model:** {config.get('eval_model', 'claude-sonnet-4-20250514')}  
            **Total Metrics:** {len(DEFAULT_METRICS) + len(st.session_state.custom_metrics)}
            """)
            
            if st.session_state.custom_metrics:
                with st.expander("View custom metrics"):
                    for metric in st.session_state.custom_metrics:
                        st.write(f"• {metric['name']}")
    
    st.markdown("### Status Checks")
    col1, col2 = st.columns(2)
    with col1:
        if rag_ready:
            st.success("✅ RAG knowledge base ready")
        else:
            st.warning("⚠️ No RAG connection. Please select a collection in RAG Documents tab")
    
    with col2:
        if eval_ready:
            st.success("✅ Evaluation data ready")
        else:
            st.error("❌ Please upload evaluation Excel file first")
    
    if rag_ready and eval_ready:
        st.markdown("### Evaluation Settings")
        
        max_questions = len(st.session_state.evaluation_data)
        
        if max_questions == 1:
            num_questions = 1
            st.info(f"Will evaluate 1 question from the dataset")
        else:
            num_questions = st.slider(
                "Number of questions to evaluate", 
                min_value=1, 
                max_value=max_questions, 
                value=min(5, max_questions),
                help="Start with fewer questions for testing"
            )
        
        if st.button("🚀 Start Evaluation", type="primary"):
            if not config["api_key"] or not config["eval_api_key"]:
                provider_name = config.get("eval_provider", "Claude")
                st.error(f"Please provide both LLM and {provider_name} API keys in the Configuration tab")
            else:
                all_metrics = DEFAULT_METRICS.copy()
                for custom_metric in st.session_state.custom_metrics:
                    all_metrics[custom_metric['id']] = {
                        "name": custom_metric['name'],
                        "description": custom_metric['description']
                    }
                
                llm = get_llm_provider(config["provider"], config["model_name"], 
                                     config["api_key"], config["temperature"])
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                df = st.session_state.evaluation_data.head(num_questions)
                
                for idx, row in df.iterrows():
                    question = row['Question']
                    ground_truth = row['Ground truth']
                    chat_history = row.get('Chat history', '')
                    
                    status_text.text(f"Processing question {idx + 1}/{len(df)}: {question[:50]}...")
                    
                    rag_chain = create_rag_chain(llm, st.session_state.vectorstore, 
                                                config["num_chunks"], chat_history)
                    
                    with st.spinner(f"Generating RAG response for question {idx + 1}..."):
                        rag_response = generate_rag_response(rag_chain, question)
                        response = rag_response['answer']
                        source_docs = rag_response['source_documents']
                    
                    retrieved_chunks = [doc.page_content for doc in source_docs]
                    
                    with st.expander(f"Generated Response for Question {idx + 1}", expanded=False):
                        st.write(f"**Question:** {question}")
                        st.write(f"**Response:** {response}")
                        if chat_history:
                            st.write(f"**Chat History:** {chat_history}")
                        st.write(f"**Retrieved {len(retrieved_chunks)} chunks**")
                    
                    eval_provider = config.get("eval_provider", "Claude")
                    if eval_provider == "Claude":
                        provider_name = "Claude"
                    elif eval_provider == "Custom Models":
                        provider_name = "Custom Models"
                    else:
                        provider_name = "OpenAI"
                    
                    # Get base URL for custom models if needed
                    base_url = None
                    if eval_provider == "Custom Models":
                        selected_model = st.session_state.get('selected_evaluation_model', None)
                        if selected_model:
                            base_url = selected_model.get('base_url')
                    
                    with st.spinner(f"Evaluating response with {provider_name} for question {idx + 1}..."):
                        evaluation = evaluate_response(
                            question, response, ground_truth, 
                            retrieved_chunks, config["eval_api_key"], all_metrics,
                            chat_history, config.get("eval_provider", "Claude"), 
                            config.get("eval_model", "claude-sonnet-4-20250514"),
                            base_url
                        )
                    
                    result = {
                        "question": question,
                        "ground_truth": ground_truth,
                        "chat_history": chat_history,
                        "response": response,
                        "retrieved_chunks": retrieved_chunks,
                        "evaluation": evaluation,
                        "model": f"{config['provider']} - {config['model_name']}",
                        "collection": st.session_state.active_collection,
                        "timestamp": datetime.now().isoformat()
                    }
                    results.append(result)
                    
                    progress_bar.progress((idx + 1) / len(df))
                
                st.session_state.evaluation_results = results
                status_text.text("✅ Evaluation completed!")
                
                st.success(f"Completed evaluation of {len(results)} questions")
                
                if results:
                    avg_scores = {}
                    for metric_key in all_metrics.keys():
                        scores = [r['evaluation']['metrics'].get(metric_key, {}).get('score', 0) 
                                for r in results]
                        avg_scores[metric_key] = np.mean(scores) if scores else 0
                    
                    st.markdown("### Average Scores")
                    cols = st.columns(4)
                    for i, (metric_key, metric_info) in enumerate(all_metrics.items()):
                        col_idx = i % 4
                        with cols[col_idx]:
                            st.metric(
                                label=metric_info['name'],
                                value=f"{avg_scores.get(metric_key, 0):.2f}/10"
                            )
