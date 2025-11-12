"""
Valuation RAG System
Retrieval-Augmented Generation system for valuation parameters and methodologies
"""

import os
from typing import List, Optional
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from vision_extractor import VisionDocumentExtractor


class ValuationRAG:
    """
    RAG system for retrieving relevant valuation parameters and methodologies
    from a reference PDF document
    """
    
    def __init__(self, valuation_pdf_path: str, 
                 embedding_model: Optional[str] = None,
                 ollama_base_url: Optional[str] = None):
        """
        Initialize the Valuation RAG system
        
        Args:
            valuation_pdf_path: Path to the PDF containing valuation parameters
            embedding_model: Name of embedding model (default: from env or nomic-embed-text)
            ollama_base_url: Ollama base URL (default: from env or host.docker.internal:11434)
        """
        
        self.valuation_pdf_path = valuation_pdf_path
        self.embedding_model_name = embedding_model or os.getenv('EMBEDDING_MODEL', 'nomic-embed-text')
        self.base_url = ollama_base_url or os.getenv('OLLAMA_BASE_URL', 'http://host.docker.internal:11434')
        
        print(f"  Initializing Valuation RAG System")
        print(f"    PDF: {valuation_pdf_path}")
        print(f"    Embedding Model: {self.embedding_model_name}")
        print(f"    Base URL: {self.base_url}")
        
        if not os.path.exists(valuation_pdf_path):
            raise FileNotFoundError(f"Valuation PDF not found: {valuation_pdf_path}")
        
        # Initialize components
        self.embeddings = OllamaEmbeddings(
            model=self.embedding_model_name,
            base_url=self.base_url
        )
        
        self.vectorstore = None
        self.documents = []
        
        # Load and index the valuation parameters
        self._load_and_index()
    
    def _load_and_index(self):
        """Load valuation PDF and create vector store index"""
        
        print("  Loading valuation parameters PDF...")
        
        # Extract content from PDF using vision model
        extractor = VisionDocumentExtractor()
        content = extractor.extract_from_pdf(self.valuation_pdf_path)
        
        print(f"  ✓ Extracted {len(content)} characters from valuation PDF")
        
        # Split content into chunks for better retrieval
        print("  Creating document chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_text(content)
        print(f"  ✓ Created {len(chunks)} chunks")
        
        # Create document objects
        self.documents = [
            Document(
                page_content=chunk,
                metadata={
                    "source": self.valuation_pdf_path,
                    "chunk_id": i
                }
            )
            for i, chunk in enumerate(chunks)
        ]
        
        # Create vector store
        print("  Building vector store index...")
        try:
            self.vectorstore = Chroma.from_documents(
                documents=self.documents,
                embedding=self.embeddings,
                collection_name="valuation_parameters",
                persist_directory=None  # In-memory only
            )
            print("  ✓ Vector store created successfully")
        except Exception as e:
            print(f"  ✗ Error creating vector store: {e}")
            raise
    
    def query(self, question: str, k: int = 5, score_threshold: float = 0.0) -> str:
        """
        Query the valuation parameters using semantic search
        
        Args:
            question: Query string
            k: Number of chunks to retrieve
            score_threshold: Minimum similarity score (0.0 to 1.0)
            
        Returns:
            Concatenated relevant text from valuation parameters
        """
        
        if not self.vectorstore:
            raise RuntimeError("Vector store not initialized. Call _load_and_index() first.")
        
        try:
            # Perform similarity search
            results = self.vectorstore.similarity_search(
                query=question,
                k=k
            )
            
            if not results:
                return "No relevant valuation parameters found."
            
            # Combine results
            retrieved_content = []
            for i, doc in enumerate(results, 1):
                retrieved_content.append(f"--- Relevant Section {i} ---")
                retrieved_content.append(doc.page_content)
                retrieved_content.append("")  # Empty line separator
            
            return "\n".join(retrieved_content)
            
        except Exception as e:
            print(f"  Warning: Query failed: {e}")
            return f"Error retrieving valuation parameters: {e}"
    
    def query_with_scores(self, question: str, k: int = 5) -> List[tuple]:
        """
        Query with similarity scores
        
        Args:
            question: Query string
            k: Number of chunks to retrieve
            
        Returns:
            List of (document, score) tuples
        """
        
        if not self.vectorstore:
            raise RuntimeError("Vector store not initialized.")
        
        try:
            results = self.vectorstore.similarity_search_with_score(
                query=question,
                k=k
            )
            return results
        except Exception as e:
            print(f"  Warning: Query with scores failed: {e}")
            return []
    
    def get_all_content(self) -> str:
        """
        Get all valuation parameter content (not recommended for large docs)
        
        Returns:
            All content from the valuation PDF
        """
        
        if not self.documents:
            return "No documents loaded"
        
        return "\n\n".join([doc.page_content for doc in self.documents])
    
    def search_by_keyword(self, keyword: str, k: int = 5) -> str:
        """
        Simple keyword-based search (less sophisticated than semantic search)
        
        Args:
            keyword: Keyword to search for
            k: Maximum number of chunks to return
            
        Returns:
            Concatenated chunks containing the keyword
        """
        
        if not self.documents:
            return "No documents loaded"
        
        matching_docs = [
            doc for doc in self.documents
            if keyword.lower() in doc.page_content.lower()
        ]
        
        if not matching_docs:
            return f"No content found containing keyword: {keyword}"
        
        # Limit results
        matching_docs = matching_docs[:k]
        
        results = []
        for i, doc in enumerate(matching_docs, 1):
            results.append(f"--- Match {i} ---")
            results.append(doc.page_content)
            results.append("")
        
        return "\n".join(results)
    
    def get_statistics(self) -> dict:
        """
        Get statistics about the indexed content
        
        Returns:
            Dictionary with statistics
        """
        
        stats = {
            "pdf_path": self.valuation_pdf_path,
            "total_chunks": len(self.documents),
            "total_characters": sum(len(doc.page_content) for doc in self.documents),
            "embedding_model": self.embedding_model_name,
            "vectorstore_initialized": self.vectorstore is not None
        }
        
        return stats


# Utility functions

def test_valuation_rag(pdf_path: str):
    """
    Test the Valuation RAG system
    
    Args:
        pdf_path: Path to valuation parameters PDF
    """
    
    print(f"\nTesting Valuation RAG System")
    print("=" * 80)
    
    # Initialize RAG
    rag = ValuationRAG(pdf_path)
    
    # Print statistics
    stats = rag.get_statistics()
    print("\nRAG Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test queries
    test_queries = [
        "What valuation methodologies should be used?",
        "What are the discount rates?",
        "What are the P/E ratio benchmarks?",
        "DCF valuation parameters"
    ]
    
    print("\n" + "=" * 80)
    print("Test Queries:")
    print("=" * 80)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 80)
        
        result = rag.query(query, k=3)
        print(result)
        print()


if __name__ == "__main__":
    # Test mode
    import sys
    
    if len(sys.argv) > 1:
        test_pdf_path = sys.argv[1]
        test_valuation_rag(test_pdf_path)
    else:
        print("Usage: python valuation_rag.py <path_to_valuation_pdf>")
        print("\nThis will test the RAG system on your valuation parameters PDF.")