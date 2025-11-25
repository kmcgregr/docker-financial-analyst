"""
Standalone tool to create vector databases from PDFs
No vision model required - uses pure text extraction
"""

import os
import sys
import argparse
from pathlib import Path
import fitz  # PyMuPDF
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text from a PDF file
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Extracted text content
    """
    print(f"\nExtracting text from: {pdf_path}")
    
    doc = fitz.open(pdf_path)
    text_content = []
    page_count = len(doc)  # Store page count before closing
    
    for page_num in range(page_count):
        page = doc[page_num]
        page_text = page.get_text()
        
        if page_text.strip():
            text_content.append(f"\n--- Page {page_num + 1} ---\n")
            text_content.append(page_text)
    
    doc.close()
    
    full_text = "\n".join(text_content)
    print(f"✓ Extracted {len(full_text)} characters from {page_count} pages")
    
    return full_text


def create_vector_database(
    pdf_path: str,
    output_dir: str,
    collection_name: str = "documents",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_model: str = None,
    ollama_base_url: str = None
):
    """
    Create a vector database from a PDF file
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save the vector database
        collection_name: Name for the collection
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        embedding_model: Ollama embedding model name
        ollama_base_url: Ollama base URL
    """
    
    # Get configuration
    embedding_model = embedding_model or os.getenv('EMBEDDING_MODEL', 'nomic-embed-text')
    ollama_base_url = ollama_base_url or os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    
    print("\n" + "="*80)
    print("PDF TO VECTOR DATABASE CONVERTER")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  PDF: {pdf_path}")
    print(f"  Output Directory: {output_dir}")
    print(f"  Collection Name: {collection_name}")
    print(f"  Embedding Model: {embedding_model}")
    print(f"  Ollama URL: {ollama_base_url}")
    print(f"  Chunk Size: {chunk_size}")
    print(f"  Chunk Overlap: {chunk_overlap}")
    
    # Step 1: Extract text
    print("\n" + "-"*80)
    print("STEP 1: Extracting Text from PDF")
    print("-"*80)
    content = extract_text_from_pdf(pdf_path)
    
    if not content or len(content.strip()) < 100:
        print("ERROR: Insufficient text extracted. PDF may be image-based or empty.")
        sys.exit(1)
    
    # Step 2: Split into chunks
    print("\n" + "-"*80)
    print("STEP 2: Splitting Text into Chunks")
    print("-"*80)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_text(content)
    print(f"✓ Created {len(chunks)} chunks")
    
    # Create document objects
    documents = [
        Document(
            page_content=chunk,
            metadata={
                "source": pdf_path,
                "chunk_id": i,
                "filename": os.path.basename(pdf_path)
            }
        )
        for i, chunk in enumerate(chunks)
    ]
    
    # Step 3: Create embeddings
    print("\n" + "-"*80)
    print("STEP 3: Creating Embeddings")
    print("-"*80)
    print("This may take a few minutes...")
    
    embeddings = OllamaEmbeddings(
        model=embedding_model,
        base_url=ollama_base_url
    )
    
    # Step 4: Create vector store
    print("\n" + "-"*80)
    print("STEP 4: Building Vector Database")
    print("-"*80)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=output_dir
    )
    
    print(f"✓ Vector database created successfully")
    print(f"✓ Saved to: {output_dir}")
    
    # Step 5: Test the database
    print("\n" + "-"*80)
    print("STEP 5: Testing Vector Database")
    print("-"*80)
    
    test_query = "What is this document about?"
    results = vectorstore.similarity_search(test_query, k=3)
    
    print(f"\nTest Query: '{test_query}'")
    print(f"Found {len(results)} relevant chunks:")
    
    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
    
    # Print statistics
    print("\n" + "="*80)
    print("COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nStatistics:")
    print(f"  Total Documents: {len(documents)}")
    print(f"  Total Characters: {sum(len(doc.page_content) for doc in documents):,}")
    print(f"  Collection Name: {collection_name}")
    print(f"  Storage Location: {output_dir}")
    
    print("\n" + "="*80)
    print("HOW TO USE YOUR VECTOR DATABASE")
    print("="*80)
    print("\nIn your Python code:")
    print(f"""
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Load the vector database
embeddings = OllamaEmbeddings(model="{embedding_model}")
vectorstore = Chroma(
    collection_name="{collection_name}",
    persist_directory="{output_dir}",
    embedding_function=embeddings
)

# Query the database
results = vectorstore.similarity_search("your query here", k=5)
for doc in results:
    print(doc.page_content)
""")


def main():
    parser = argparse.ArgumentParser(
        description="Create a vector database from a PDF file (no vision model needed)"
    )
    parser.add_argument(
        "pdf_path",
        help="Path to the PDF file"
    )
    parser.add_argument(
        "--output-dir",
        default="./vectordb",
        help="Output directory for vector database (default: ./vectordb)"
    )
    parser.add_argument(
        "--collection-name",
        default="documents",
        help="Name for the vector database collection (default: documents)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Size of text chunks (default: 1000)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between chunks (default: 200)"
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Ollama embedding model (default: from env or nomic-embed-text)"
    )
    parser.add_argument(
        "--ollama-url",
        default=None,
        help="Ollama base URL (default: from env or http://localhost:11434)"
    )
    
    args = parser.parse_args()
    
    # Validate PDF exists
    if not os.path.exists(args.pdf_path):
        print(f"ERROR: PDF file not found: {args.pdf_path}")
        sys.exit(1)
    
    try:
        create_vector_database(
            pdf_path=args.pdf_path,
            output_dir=args.output_dir,
            collection_name=args.collection_name,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            embedding_model=args.embedding_model,
            ollama_base_url=args.ollama_url
        )
    except Exception as e:
        print(f"\n{'='*80}")
        print("ERROR")
        print(f"{'='*80}")
        print(f"\n{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()