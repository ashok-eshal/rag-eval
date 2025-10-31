"""RAG Generation Tab UI"""

import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import datetime

from src.config.session import load_config
from src.utils.helpers import create_sample_questions_data
from src.services.llm_service import get_llm_provider
from src.services.rag_service import create_rag_chain, generate_rag_response


def render_rag_gen_tab():
    """Render the RAG Generation tab"""
    st.header("🤖 RAG Generation")
    st.markdown("Upload an Excel file with questions and generate answers using your RAG system.")
    
    rag_ready = st.session_state.vectorstore is not None
    config = load_config()
    
    if not rag_ready:
        st.warning("⚠️ No RAG connection. Please select a collection in the RAG Documents tab first.")
    else:
        st.success(f"✅ Connected to collection: `{st.session_state.active_collection}`")
        
        # Sample template download
        st.subheader("📝 Sample Template")
        st.info("Download this sample file to understand the required format. Only 'Question' column is required.")
        
        sample_df = create_sample_questions_data()
        st.dataframe(sample_df)
        
        sample_output = BytesIO()
        with pd.ExcelWriter(sample_output, engine='openpyxl') as writer:
            sample_df.to_excel(writer, index=False, sheet_name='Questions')
        
        st.download_button(
            label="📥 Download Sample Excel Template",
            data=sample_output.getvalue(),
            file_name="rag_generation_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        st.markdown("---")
        
        # File upload
        st.subheader("📤 Upload Questions File")
        st.markdown("Upload Excel file with 'Question' column (other columns are optional and will be preserved)")
        
        questions_file = st.file_uploader(
            "Choose Excel file with questions",
            type=['xlsx', 'xls'],
            key="rag_gen_upload"
        )
        
        if questions_file is not None:
            try:
                df = pd.read_excel(questions_file)
                
                if 'Question' not in df.columns:
                    st.error("❌ Excel file must contain a 'Question' column")
                    st.info(f"Found columns: {', '.join(df.columns)}")
                else:
                    st.success(f"✅ File uploaded successfully! Found {len(df)} questions.")
                    st.dataframe(df.head(10))
                    
                    if len(df) > 10:
                        st.info(f"Showing first 10 rows. Total rows: {len(df)}")
                    
                    # Note about Chat history column
                    has_chat_history = 'Chat history' in df.columns
                    if has_chat_history:
                        st.info("✅ Found 'Chat history' column - will be used for context-aware responses (input). Generated answers will be saved in 'Chat history' column in output.")
                        df['Chat history'] = df['Chat history'].fillna('')
                    else:
                        st.info("ℹ️ No 'Chat history' column found - answers will be generated without prior context")
                        df['Chat history'] = ''
                    
                    # Processing settings
                    st.markdown("---")
                    st.subheader("⚙️ Processing Settings")
                    
                    if len(df) == 1:
                        num_questions = 1
                        st.info(f"Will process 1 question")
                    else:
                        num_questions = st.slider(
                            "Number of questions to process", 
                            min_value=1, 
                            max_value=len(df), 
                            value=min(10, len(df)),
                            help="Start with fewer questions for testing"
                        )
                    
                    # Process button
                    if st.button("🚀 Generate Answers", type="primary"):
                        if not config["api_key"]:
                            st.error("Please provide LLM API key in the Configuration tab")
                        else:
                            # Initialize LLM
                            llm = get_llm_provider(
                                config["provider"], 
                                config["model_name"], 
                                config["api_key"], 
                                config["temperature"]
                            )
                            
                            # Process questions
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            df_to_process = df.head(num_questions).copy()
                            
                            # Store input chat history separately before overwriting
                            input_chat_history = df_to_process.get('Chat history', pd.Series([''] * len(df_to_process)))
                            
                            # Initialize Chat history column for output (will contain generated answers)
                            df_to_process['Chat history'] = ''
                            
                            for idx, row in df_to_process.iterrows():
                                question = row['Question']
                                chat_history_input = input_chat_history.iloc[idx] if idx < len(input_chat_history) else ''
                                
                                status_text.text(f"Processing question {idx + 1}/{len(df_to_process)}: {question[:50]}...")
                                
                                # Create RAG chain for this question
                                rag_chain = create_rag_chain(
                                    llm, 
                                    st.session_state.vectorstore, 
                                    config["num_chunks"], 
                                    chat_history_input
                                )
                                
                                # Generate response
                                with st.spinner(f"Generating answer for question {idx + 1}..."):
                                    rag_response = generate_rag_response(rag_chain, question)
                                    answer = rag_response['answer']
                                
                                # Store result in Chat history column
                                df_to_process.at[idx, 'Chat history'] = answer
                                
                                # Show preview
                                with st.expander(f"Question {idx + 1}: {question[:80]}...", expanded=False):
                                    st.markdown(f"**Question:** {question}")
                                    if chat_history_input:
                                        st.markdown(f"**Input Chat History:** {chat_history_input}")
                                    st.markdown(f"**Generated Answer:**")
                                    st.write(answer)
                                
                                progress_bar.progress((idx + 1) / len(df_to_process))
                            
                            status_text.text("✅ Generation completed!")
                            st.success(f"Completed generating answers for {len(df_to_process)} questions")
                            
                            # Prepare output with only Question and Chat history columns
                            output_df = df_to_process[['Question', 'Chat history']].copy()
                            
                            # Store results in session state
                            st.session_state.rag_generation_results = output_df
                            
                            # Prepare download
                            st.markdown("---")
                            st.subheader("📥 Download Results")
                            
                            output = BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                output_df.to_excel(writer, index=False, sheet_name='Questions and Answers')
                            
                            st.download_button(
                                label="📥 Download Excel with Answers",
                                data=output.getvalue(),
                                file_name=f"rag_generated_answers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                            
                            # Display summary
                            st.markdown("### 📊 Summary")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Questions", len(output_df))
                            with col2:
                                avg_answer_length = output_df['Chat history'].str.len().mean()
                                st.metric("Avg Answer Length", f"{avg_answer_length:.0f} chars")
                            with col3:
                                completed = len(output_df[output_df['Chat history'] != ''])
                                st.metric("Completed", f"{completed}/{len(output_df)}")
                            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.exception(e)
