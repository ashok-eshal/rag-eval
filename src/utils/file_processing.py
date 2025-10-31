"""File processing utilities for document extraction"""

import streamlit as st
import PyPDF2
import docx
from pathlib import Path
import os
import tempfile


def extract_text_from_pdf(file_path: str, use_ocr: bool = False, ocr_model_info: dict = None) -> str:
    """Extract text from PDF file, optionally using OCR for scanned documents"""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        
        # If OCR is enabled and text extraction failed, use OCR
        if use_ocr and ocr_model_info:
            # Check if PDF is likely scanned (little or no text extracted)
            if len(text.strip()) < 50:  # Very minimal text suggests scanned PDF
                st.info(f"Detected scanned PDF. Using OCR for text extraction...")
                from src.services.ocr_service import extract_text_with_ocr
                text = extract_text_with_ocr(file_path, ocr_model_info)
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
    return text


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX file"""
    text = ""
    try:
        doc = docx.Document(file_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
    return text


def extract_text_from_txt(file_path: str) -> str:
    """Extract text from TXT file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        st.error(f"Error reading TXT: {str(e)}")
        return ""


def process_uploaded_file(uploaded_file, use_ocr: bool = False) -> str:
    """Process uploaded file and extract text"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name
    
    file_extension = Path(uploaded_file.name).suffix.lower()
    
    if file_extension == '.pdf':
        # Check if OCR model is configured and should be used
        ocr_model_info = None
        if use_ocr and st.session_state.get('selected_ocr_model'):
            ocr_model_info = st.session_state.selected_ocr_model
        text = extract_text_from_pdf(tmp_path, use_ocr=use_ocr, ocr_model_info=ocr_model_info)
    elif file_extension == '.docx':
        text = extract_text_from_docx(tmp_path)
    elif file_extension in ['.txt', '.md']:
        text = extract_text_from_txt(tmp_path)
    else:
        text = ""
        st.error(f"Unsupported file type: {file_extension}")
    
    os.unlink(tmp_path)
    return text
