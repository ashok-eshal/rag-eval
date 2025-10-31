"""Results Tab UI"""

import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import datetime

from src.config.constants import DEFAULT_METRICS


def render_results_tab():
    """Render the Results tab"""
    st.header("Evaluation Results")
    
    if st.session_state.evaluation_results:
        col1, col2, col3 = st.columns(3)
        with col1:
            min_score = st.slider("Minimum Overall Score", 0.0, 10.0, 0.0, 0.5)
        with col2:
            sort_by = st.selectbox("Sort by", ["Overall Score", "Question", "Timestamp"])
        with col3:
            show_chunks = st.checkbox("Show Retrieved Chunks", value=False)
            show_chat_history = st.checkbox("Show Chat History", value=False)
        
        sorted_results = st.session_state.evaluation_results.copy()
        if sort_by == "Overall Score":
            sorted_results.sort(key=lambda x: x['evaluation'].get('overall_score', 0), reverse=True)
        elif sort_by == "Timestamp":
            sorted_results.sort(key=lambda x: x['timestamp'], reverse=True)
        
        for i, result in enumerate(sorted_results):
            overall_score = result['evaluation'].get('overall_score', 0)
            
            if overall_score >= min_score:
                with st.expander(f"Question {i+1}: {result['question'][:100]}... (Score: {overall_score:.2f})"):
                    st.caption(f"Collection: {result.get('collection', 'Unknown')}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Question:**")
                        st.write(result['question'])
                        
                        st.markdown("**Ground Truth:**")
                        st.write(result['ground_truth'])
                        
                        if show_chat_history and result.get('chat_history'):
                            st.markdown("**Chat History:**")
                            st.write(result['chat_history'])
                        
                        st.markdown("**Model Response:**")
                        st.write(result['response'])
                    
                    with col2:
                        st.markdown("**Evaluation Scores:**")
                        metrics = result['evaluation'].get('metrics', {})
                        for metric_key, metric_data in metrics.items():
                            score = metric_data.get('score', 0)
                            explanation = metric_data.get('explanation', '')
                            
                            all_metrics = DEFAULT_METRICS.copy()
                            for custom_metric in st.session_state.custom_metrics:
                                all_metrics[custom_metric['id']] = {
                                    "name": custom_metric['name'],
                                    "description": custom_metric['description']
                                }
                            
                            metric_name = all_metrics.get(metric_key, {}).get('name', metric_key)
                            
                            st.markdown(f"**{metric_name}:** {score}/10")
                            with st.container():
                                st.caption(explanation)
                    
                    st.markdown("**Summary:**")
                    st.info(result['evaluation'].get('summary', ''))
                    
                    st.markdown("**Recommendations:**")
                    st.warning(result['evaluation'].get('recommendations', ''))
                    
                    if show_chunks and result['retrieved_chunks']:
                        st.markdown("**Retrieved Chunks:**")
                        for j, chunk in enumerate(result['retrieved_chunks']):
                            with st.expander(f"Chunk {j+1}"):
                                st.text(chunk)
        
        st.markdown("### Export Results")
        if st.button("📥 Export Results to Excel"):
            export_data = []
            for result in st.session_state.evaluation_results:
                row = {
                    "Question": result['question'],
                    "Ground Truth": result['ground_truth'],
                    "Chat History": result.get('chat_history', ''),
                    "Model Response": result['response'],
                    "Overall Score": result['evaluation'].get('overall_score', 0),
                    "Model": result['model'],
                    "Collection": result.get('collection', ''),
                    "Timestamp": result['timestamp']
                }
                
                metrics = result['evaluation'].get('metrics', {})
                for metric_key, metric_data in metrics.items():
                    all_metrics = DEFAULT_METRICS.copy()
                    for custom_metric in st.session_state.custom_metrics:
                        all_metrics[custom_metric['id']] = {
                            "name": custom_metric['name'],
                            "description": custom_metric['description']
                        }
                    metric_name = all_metrics.get(metric_key, {}).get('name', metric_key)
                    row[f"{metric_name} Score"] = metric_data.get('score', 0)
                    row[f"{metric_name} Explanation"] = metric_data.get('explanation', '')
                
                export_data.append(row)
            
            export_df = pd.DataFrame(export_data)
            
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                export_df.to_excel(writer, index=False, sheet_name='Evaluation Results')
            
            st.download_button(
                label="Download Excel File",
                data=output.getvalue(),
                file_name=f"llm_evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.info("No evaluation results available. Please run evaluation first.")
