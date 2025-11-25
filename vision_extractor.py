"""
Vision Document Extractor
Extracts text and data from financial PDF documents using vision models
"""

import os
import base64
import requests
import json
from typing import Optional
from langchain_community.llms import Ollama
from utils import check_model_availability

# Try to import PyMuPDF at module level
try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    fitz = None
    FITZ_AVAILABLE = False


class VisionDocumentExtractor:
    """
    Extracts information from PDF documents using vision-language models.
    Supports both Qwen2-VL and Llama 3.2 Vision models.
    """
    
    def __init__(self, model_name: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the vision extractor
        
        Args:
            model_name: Name of the vision model (default: from env or llama3.2-vision:11b)
            base_url: Ollama base URL (default: from env or http://localhost:11434)
        """
        
        self.model_name = model_name or os.getenv('VISION_MODEL', 'llama3.2-vision:11b')
        self.base_url = base_url or os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        
        print(f"  Initializing Vision Extractor")
        print(f"    Model: {self.model_name}")
        print(f"    Base URL: {self.base_url}")
        
        # Check if PyMuPDF is available
        if not FITZ_AVAILABLE:
            print(f"    WARNING: PyMuPDF (fitz) not available. PDF extraction may fail.")
        
        # Check if the model is available
        check_model_availability(self.model_name)
        
        # Initialize Ollama LLM for text-only processing
        self.llm = Ollama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=0.0  # Deterministic for document extraction
        )
        
        # Detect if this is a vision-capable model
        self.is_vision_model = self._is_vision_capable()
        print(f"    Vision capabilities: {'Yes' if self.is_vision_model else 'No'}")
    
    def _is_vision_capable(self) -> bool:
        """Check if the model supports vision/image input"""
        vision_keywords = ['vision', 'llava', 'qwen2-vl', 'llama3.2-vision']
        return any(keyword in self.model_name.lower() for keyword in vision_keywords)
    
    def _call_ollama_api_with_image(self, prompt: str, image_base64: str) -> str:
        """
        Call Ollama API directly with image support
        
        Args:
            prompt: Text prompt for the model
            image_base64: Base64-encoded image
            
        Returns:
            Model response text
        """
        
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [image_base64],
            "stream": False,
            "options": {
                "temperature": 0.0
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()
            return result.get('response', '')
        except requests.exceptions.RequestException as e:
            print(f"      Error calling Ollama API: {e}")
            return ""
    
    def extract_from_pdf(self, pdf_path: str) -> str:
        """
        Extract all relevant information from a PDF document
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content with financial data and analysis
        """
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if not FITZ_AVAILABLE:
            raise ImportError(
                "PyMuPDF is required for PDF processing. "
                "Install with: pip install PyMuPDF"
            )
        
        try:
            # Open PDF document
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            
            print(f"    Processing {total_pages} pages...")
            
            extracted_content = []
            extracted_content.append(f"DOCUMENT: {os.path.basename(pdf_path)}")
            extracted_content.append(f"Total Pages: {total_pages}")
            extracted_content.append("=" * 70)
            
            # Process each page
            for page_num in range(total_pages):
                print(f"      Processing page {page_num + 1}/{total_pages}...")
                
                page_content = self._extract_page(doc, page_num)
                
                if page_content:
                    extracted_content.append(f"\n--- PAGE {page_num + 1} ---")
                    extracted_content.append(page_content)
            
            doc.close()
            
            final_content = "\n".join(extracted_content)
            print(f"    ✓ Extracted {len(final_content)} characters")
            
            return final_content
            
        except Exception as e:
            print(f"    ✗ Error extracting from PDF: {e}")
            raise
    
    def _extract_page(self, doc, page_num: int) -> str:
        """
        Extract content from a single page
        
        Args:
            doc: PyMuPDF document object
            page_num: Page number (0-indexed)
            
        Returns:
            Extracted text content from the page
        """
        
        try:
            page = doc[page_num]
            
            # First try text extraction (faster)
            text_content = page.get_text()
            
            # If vision model is available, use image-based extraction
            if self.is_vision_model:
                return self._vision_extract_page(page, page_num, text_content)
            else:
                # Fall back to text-based extraction with LLM enhancement
                if text_content and len(text_content.strip()) > 100:
                    return self._enhance_with_llm(text_content, page_num)
                else:
                    return text_content
                
        except Exception as e:
            print(f"      Warning: Could not process page {page_num + 1}: {e}")
            return ""
    
    def _enhance_with_llm(self, text_content: str, page_num: int) -> str:
        """
        Enhance extracted text with LLM analysis (non-vision)
        
        Args:
            text_content: Text extracted from page
            page_num: Page number
            
        Returns:
            Enhanced content with LLM analysis
        """
        
        prompt = f"""Analyze this financial document page content and extract structured information.

Raw text from page:
{text_content[:3000]}

Extract and organize:
1. All numerical financial data (revenue, profit, expenses, etc.)
2. Key financial metrics and ratios
3. Important dates and periods
4. Company information
5. Any tables or structured data

Format your response as clear, organized text with all numbers and their labels."""

        try:
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            print(f"      Warning: LLM enhancement failed for page {page_num + 1}, using text: {e}")
            return text_content
    
    def _vision_extract_page(self, page, page_num: int, fallback_text: str = "") -> str:
        """
        Extract content using vision model with image input
        
        Args:
            page: PyMuPDF page object
            page_num: Page number
            fallback_text: Text to use if vision extraction fails
            
        Returns:
            Extracted content via vision analysis
        """
        
        if not FITZ_AVAILABLE:
            print(f"      Warning: PyMuPDF not available, cannot extract page {page_num + 1}")
            return fallback_text
        
        try:
            # Render page to high-quality image
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Convert to base64
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            
            # Create vision prompt
            prompt = """You are analyzing a financial document page image. Extract ALL relevant information you can see including:

1. FINANCIAL FIGURES
   - All revenue, sales, income numbers
   - All expenses, costs
   - Profits, margins, percentages
   - Cash flow data
   - Balance sheet items
   - Year/quarter labels for all numbers

2. KEY METRICS
   - Growth rates
   - Financial ratios (P/E, ROE, margins, debt ratios, etc.)
   - Per-share data (EPS, dividends, book value, etc.)
   - KPIs and performance indicators

3. TEXTUAL INFORMATION
   - Company name and ticker
   - Time periods (Q1 2024, FY2023, etc.)
   - Section headers and titles
   - Business descriptions
   - Risk factors or important notes

4. TABLES AND CHARTS
   - Extract all data from tables with row and column labels
   - Describe trends visible in charts
   - Note any footnotes or references

5. STRUCTURE
   - Identify if this is: income statement, balance sheet, cash flow, or narrative section
   - Note any comparative periods (current vs prior year)

Be EXTREMELY thorough and precise. Extract every number with its label and context. Format as organized, structured text."""

            # Call Ollama API directly with image
            response = self._call_ollama_api_with_image(prompt, img_base64)
            
            if response and len(response.strip()) > 50:
                return response
            else:
                print(f"      Warning: Vision extraction returned minimal content, using fallback")
                return fallback_text if fallback_text else "No content extracted from this page."
            
        except Exception as e:
            print(f"      Warning: Vision extraction failed for page {page_num + 1}: {e}")
            return fallback_text if fallback_text else ""
    
    def extract_from_multiple_pdfs(self, pdf_paths: list) -> dict:
        """
        Extract content from multiple PDF files
        
        Args:
            pdf_paths: List of paths to PDF files
            
        Returns:
            Dictionary mapping filenames to extracted content
        """
        
        results = {}
        
        for pdf_path in pdf_paths:
            filename = os.path.basename(pdf_path)
            print(f"\n  Extracting {filename}...")
            
            try:
                content = self.extract_from_pdf(pdf_path)
                results[filename] = content
            except Exception as e:
                print(f"  ✗ Failed to extract {filename}: {e}")
                results[filename] = f"ERROR: Could not extract content from {filename}"
        
        return results


# Utility functions

def test_vision_extractor(pdf_path: str):
    """
    Test the vision extractor on a single PDF
    
    Args:
        pdf_path: Path to test PDF file
    """
    
    print(f"\nTesting Vision Extractor on: {pdf_path}")
    print("=" * 80)
    
    extractor = VisionDocumentExtractor()
    content = extractor.extract_from_pdf(pdf_path)
    
    print("\nExtracted Content:")
    print("=" * 80)
    print(content)
    print("=" * 80)
    print(f"\nTotal characters extracted: {len(content)}")


if __name__ == "__main__":
    # Test mode
    import sys
    
    if len(sys.argv) > 1:
        test_pdf_path = sys.argv[1]
        test_vision_extractor(test_pdf_path)
    else:
        print("Usage: python vision_extractor.py <path_to_pdf>")
        print("\nThis will test the vision extractor on a single PDF file.")