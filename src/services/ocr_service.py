"""OCR service for extracting text from scanned PDFs and images"""

import streamlit as st
import base64
import io
from typing import Optional, List
import PyPDF2
from PIL import Image

try:
    from openai import OpenAI
    OPENAI_CLIENT_AVAILABLE = True
except Exception:
    OPENAI_CLIENT_AVAILABLE = False

try:
    from pdf2image import convert_from_path, convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False


def is_scanned_pdf(file_path: str) -> bool:
    """Detect if PDF is a scanned document by checking if it contains extractable text"""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_chars = 0
            num_pages = len(pdf_reader.pages)
            
            # Sample first few pages to check if text exists
            pages_to_check = min(3, num_pages)
            for i in range(pages_to_check):
                try:
                    text = pdf_reader.pages[i].extract_text()
                    if text:
                        total_chars += len(text.strip())
                except:
                    pass
            
            # If very few characters extracted, likely scanned
            avg_chars_per_page = total_chars / pages_to_check if pages_to_check > 0 else 0
            return avg_chars_per_page < 50
    except Exception as e:
        st.warning(f"Could not determine if PDF is scanned: {str(e)}")
        return False


def extract_text_with_ocr(pdf_path: str, model_info: dict) -> str:
    """Extract text from PDF using OCR model"""
    try:
        # Check if ocr library is available
        if not PDF2IMAGE_AVAILABLE:
            st.error("pdf2image library not installed. Install with: pip install pdf2image")
            # On Windows, also need: Install poppler from https://github.com/oschwartz10612/poppler-windows/releases
            return ""
        
        # Convert PDF pages to images
        try:
            images = convert_from_bytes(open(pdf_path, 'rb').read())
        except Exception as pdf_error:
            # Check if it's a poppler error
            error_msg = str(pdf_error).lower()
            if "poppler" in error_msg or "pdftoppm" in error_msg or "convert" in error_msg:
                st.error("""
                **Poppler is not installed or not accessible!**
                
                Please install Poppler:
                - **Docker**: Poppler is installed in the Docker image
                - **Linux**: `sudo apt-get install poppler-utils`
                - **macOS**: `brew install poppler`
                - **Windows**: Download from https://github.com/oschwartz10612/poppler-windows/releases
                
                Then ensure Poppler is in your system PATH.
                """)
                return ""
            else:
                # Try fallback
                st.warning(f"Could not convert PDF with convert_from_bytes: {str(pdf_error)}")
                try:
                    images = convert_from_path(pdf_path)
                except Exception as path_error:
                    st.error(f"Failed to convert PDF to images: {str(path_error)}")
                    return ""
        
        # Get OCR model configuration
        model_name = model_info.get('name', '')
        api_key = model_info.get('api_key', '')
        base_url = model_info.get('base_url', 'https://api.deepinfra.com/v1/openai')
        
        if not OPENAI_CLIENT_AVAILABLE:
            st.error("OpenAI library not available for OCR")
            return ""
        
        # Initialize OpenAI client
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        all_text = []
        
        for i, image in enumerate(images):
            # Convert image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Prepare the prompt for OCR (vision models only for OCR)
            prompt = """Please extract all text from this image. Return only the extracted text without any additional formatting or explanation."""
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4096
            )
            
            extracted_text = response.choices[0].message.content
            all_text.append(extracted_text)
            # st.info(f"Extracted text from page {i+1}/{len(images)}")
        
        return "\n\n".join(all_text)
        
    except Exception as e:
        st.error(f"Error during OCR extraction: {str(e)}")
        return ""


def pdf_to_images(pdf_path: str) -> List[Image.Image]:
    """Convert PDF to list of PIL Images"""
    try:
        if not PDF2IMAGE_AVAILABLE:
            st.error("pdf2image library not installed. Install with: pip install pdf2image")
            return []
        with open(pdf_path, 'rb') as file:
            pdf_bytes = file.read()
            images = convert_from_bytes(pdf_bytes)
        return images
    except Exception as e:
        st.error(f"Error converting PDF to images: {str(e)}")
        return []


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

