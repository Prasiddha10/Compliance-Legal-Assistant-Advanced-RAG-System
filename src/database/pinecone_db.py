"""Pinecone vector database implementation."""
import time
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from src.utils.embeddings import SentenceTransformerEmbeddings
from src.config import Config
import logging

logger = logging.getLogger(__name__)

class PineconeDBManager:
    """Manage Pinecone vector database operations."""
    
    def __init__(self, index_name: Optional[str] = None):
        self.index_name = index_name or Config.PINECONE_INDEX_NAME
        self.embeddings = SentenceTransformerEmbeddings(Config.EMBEDDING_MODEL)
        
        # Initialize Pinecone
        if not Config.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY is required")
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        
        # Check if index exists, create if not
        self._ensure_index_exists()
        
        # Get the index
        self.index = self.pc.Index(self.index_name)
        
        # Initialize Langchain Pinecone vectorstore with proper API key
        self.vectorstore = LangchainPinecone.from_existing_index(
            index_name=self.index_name,
            embedding=self.embeddings
        )
        
        logger.info(f"Pinecone initialized with index: {self.index_name}")
    
    def _ensure_index_exists(self):
        """Ensure the Pinecone index exists."""
        try:
            # Get list of existing indexes
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                # Create new index with correct embedding dimension
                self.pc.create_index(
                    name=self.index_name,
                    dimension=Config.EMBEDDING_DIMENSION,  # Use configured dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=getattr(Config, "PINECONE_REGION", "us-east-1")
                    )
                )
                
                # Wait for index to be ready
                while self.index_name not in [index.name for index in self.pc.list_indexes()]:
                    time.sleep(1)
                
                logger.info(f"Created new Pinecone index: {self.index_name}")
            else:
                # Check if existing index dimension matches our embedding dimension
                index_info = self.pc.describe_index(self.index_name)
                existing_dimension = index_info.dimension
                expected_dimension = Config.EMBEDDING_DIMENSION
                
                if existing_dimension != expected_dimension:
                    logger.warning(
                        f"Dimension mismatch! Pinecone index '{self.index_name}' has dimension {existing_dimension}, "
                        f"but embedding model produces {expected_dimension} dimensions. "
                        f"You may need to delete and recreate the index or use a different embedding model."
                    )
                    # Optionally, we could automatically recreate the index here
                    # For now, just log the warning and continue (will fail on add_documents)
                
                logger.info(f"Using existing Pinecone index: {self.index_name} (dimension: {existing_dimension})")
                
        except Exception as e:
            logger.error(f"Error ensuring Pinecone index exists: {e}")
            raise
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector database."""
        try:
            # Convert documents to texts and metadatas
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            # Add documents to vectorstore
            doc_ids = self.vectorstore.add_texts(texts, metadatas)
            
            logger.info(f"Added {len(documents)} documents to Pinecone")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error adding documents to Pinecone: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 5, 
                         filter_metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Perform similarity search."""
        try:
            if filter_metadata:
                results = self.vectorstore.similarity_search(
                    query, k=k, filter=filter_metadata
                )
            else:
                results = self.vectorstore.similarity_search(query, k=k)
            
            logger.info(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            raise
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """Perform similarity search with relevance scores."""
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            logger.info(f"Found {len(results)} similar documents with scores")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search with scores: {e}")
            raise
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the Pinecone index."""
        try:
            stats = self.index.describe_index_stats()
            
            return {
                "index_name": self.index_name,
                "total_vector_count": stats.get("total_vector_count", 0),
                "dimension": stats.get("dimension", 384),
                "index_fullness": stats.get("index_fullness", 0.0)
            }
            
        except Exception as e:
            logger.warning(f"Could not get index stats: {e}")
            return {
                "index_name": self.index_name,
                "total_vector_count": 0,
                "dimension": 384,
                "index_fullness": 0.0
            }
    
    def delete_index(self):
        """Delete the entire Pinecone index."""
        try:
            self.pc.delete_index(self.index_name)
            logger.info(f"Deleted Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Error deleting index: {e}")
            raise
    
    def delete_vectors(self, ids: List[str]):
        """Delete specific vectors by ID."""
        try:
            self.index.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} vectors")
            
        except Exception as e:
            logger.error(f"Error deleting vectors: {e}")
            raise
    
    def test_pinecone_connection(self) -> Dict[str, Any]:
        """Test Pinecone connection and document indexing."""
        try:
            # Test 1: Check if we can get index stats
            stats = self.get_index_stats()
            
            # Test 2: Try to add a test document
            test_doc = Document(
                page_content="This is a test document for Pinecone connectivity.",
                metadata={"test": True, "source": "test"}
            )
            
            doc_ids = self.add_documents([test_doc])
            
            # Test 3: Try to search for the test document
            results = self.similarity_search("test document", k=1)
            
            # Test 4: Clean up the test document
            if doc_ids:
                self.delete_vectors(doc_ids)
            
            return {
                "connection": "success",
                "index_stats": stats,
                "test_document_added": len(doc_ids) > 0,
                "search_results": len(results),
                "message": "Pinecone is working correctly"
            }
            
        except Exception as e:
            logger.error(f"Pinecone connection test failed: {e}")
            return {
                "connection": "failed",
                "error": str(e),
                "message": "Pinecone connection test failed"
            }
