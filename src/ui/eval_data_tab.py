"""Evaluation Data Tab UI"""

import streamlit as st
import pandas as pd
from io import BytesIO
from src.utils.helpers import create_sample_evaluation_data


def render_eval_data_tab():
    """Render the Evaluation Data tab"""
    st.header("Upload Evaluation Data")
    
    st.subheader("📝 Sample Evaluation File")
    st.info("Download this sample file to understand the required format. You can modify it with your own data and upload it back.")
    
    sample_df = create_sample_evaluation_data()
    
    st.markdown("**Preview of sample data:**")
    st.dataframe(sample_df)
    
    sample_output = BytesIO()
    with pd.ExcelWriter(sample_output, engine='openpyxl') as writer:
        sample_df.to_excel(writer, index=False, sheet_name='Evaluation Data')
    
    st.download_button(
        label="📥 Download Sample Excel Template",
        data=sample_output.getvalue(),
        file_name="evaluation_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    st.markdown("---")
    
    st.subheader("📤 Upload Your Evaluation Data")
    st.markdown("Upload Excel file with 'Question', 'Ground truth', and optional 'Chat history' columns")
    
    eval_file = st.file_uploader(
        "Choose evaluation Excel file",
        type=['xlsx', 'xls']
    )
    
    if eval_file is not None:
        df = pd.read_excel(eval_file)
        
        required_columns = ['Question', 'Ground truth']
        if all(col in df.columns for col in required_columns):
            st.success("✅ Evaluation file uploaded successfully!")
            
            has_chat_history = 'Chat history' in df.columns
            if has_chat_history:
                st.info("✅ Found 'Chat history' column - will be used for engagement evaluation")
                df['Chat history'] = df['Chat history'].fillna('')
            else:
                st.info("ℹ️ No 'Chat history' column found - proceeding without chat history context")
                df['Chat history'] = ''
            
            st.dataframe(df)
            
            st.session_state.evaluation_data = df
            
            st.markdown("### Evaluation Data Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Questions", len(df))
            with col2:
                st.metric("Avg Question Length", f"{df['Question'].str.len().mean():.0f} chars")
            with col3:
                st.metric("Avg Ground Truth Length", f"{df['Ground truth'].str.len().mean():.0f} chars")
            with col4:
                if has_chat_history:
                    non_empty_history = df[df['Chat history'] != '']['Chat history']
                    if len(non_empty_history) > 0:
                        st.metric("Questions with History", f"{len(non_empty_history)}/{len(df)}")
                    else:
                        st.metric("Questions with History", "0")
        else:
            st.error(f"❌ Excel file must contain columns: {required_columns}")
            st.info("Found columns: " + ", ".join(df.columns))
            st.warning("Please use the sample template provided above to ensure correct column names.")
