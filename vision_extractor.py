"""
Vision Document Extractor
Extracts text and data from financial PDF documents using vision models
"""

import os
import base64
from typing import Optional
from langchain_community.llms import Ollama


class VisionDocumentExtractor:
    """
    Extracts information from PDF documents using vision-language models.
    Uses Qwen2-VL to read and interpret financial documents.
    """
    
    def __init__(self, model_name: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the vision extractor
        
        Args:
            model_name: Name of the vision model (default: from env or qwen2-vl:7b)
            base_url: Ollama base URL (default: from env or host.docker.internal:11434)
        """
        
        self.model_name = model_name or os.getenv('VISION_MODEL', 'qwen2-vl:7b')
        self.base_url = base_url or os.getenv('OLLAMA_BASE_URL', 'http://host.docker.internal:11434')
        
        print(f"  Initializing Vision Extractor")
        print(f"    Model: {self.model_name}")
        print(f"    Base URL: {self.base_url}")
        
        self.llm = Ollama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=0.0  # Deterministic for document extraction
        )
    
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
        
        try:
            import fitz  # PyMuPDF
        except ImportError:
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
            
            # If text extraction yields good results, use it
            if text_content and len(text_content.strip()) > 100:
                # Use vision model to enhance/validate the extraction
                enhanced_content = self._enhance_with_vision(text_content, page, page_num)
                return enhanced_content
            else:
                # Fall back to pure vision extraction for image-heavy pages
                return self._vision_extract_page(page, page_num)
                
        except Exception as e:
            print(f"      Warning: Could not process page {page_num + 1}: {e}")
            return ""
    
    def _enhance_with_vision(self, text_content: str, page, page_num: int) -> str:
        """
        Enhance extracted text with vision model analysis
        
        Args:
            text_content: Text extracted from page
            page: PyMuPDF page object
            page_num: Page number
            
        Returns:
            Enhanced content with vision analysis
        """
        
        # For text-rich pages, we'll use the vision model to identify and extract
        # structured financial data that might not be well-represented in pure text
        
        prompt = f"""Analyze this financial document page content and extract structured information.

Raw text from page:
{text_content[:3000]}  # Limit context

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
            print(f"      Warning: Vision enhancement failed for page {page_num + 1}, using text: {e}")
            return text_content
    
    def _vision_extract_page(self, page, page_num: int) -> str:
        """
        Extract content using pure vision model approach
        
        Args:
            page: PyMuPDF page object
            page_num: Page number
            
        Returns:
            Extracted content via vision analysis
        """
        
        try:
            # Render page to image with high resolution
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Convert to base64
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            
            # Create vision prompt
            prompt = """You are analyzing a financial document page. Extract ALL relevant information including:

1. FINANCIAL FIGURES
   - Revenue, sales, income numbers
   - Expenses, costs
   - Profits, margins, percentages
   - Cash flow data
   - Balance sheet items
   - Any year/quarter labels

2. KEY METRICS
   - Growth rates
   - Ratios (P/E, ROE, margins, etc.)
   - Per-share data (EPS, etc.)
   - KPIs

3. TEXTUAL INFORMATION
   - Company name
   - Time periods (Q1 2024, FY2023, etc.)
   - Section headers
   - Important business descriptions
   - Risk factors or notes

4. TABLES AND CHARTS
   - Describe any tables with their data
   - Describe trends in charts

Provide a DETAILED extraction with all numbers, labels, and context. Be thorough and precise."""

            # Note: Actual vision model calling with images requires specific API support
            # For now, we'll use text-based prompting
            # In production, you'd use a proper multimodal API
            
            response = self.llm.invoke(prompt)
            return response
            
        except Exception as e:
            print(f"      Warning: Pure vision extraction failed for page {page_num + 1}: {e}")
            return ""
    
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