"""ChromaDB vector database implementation."""
import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from src.utils.embeddings import SentenceTransformerEmbeddings
from src.config import Config
import logging

logger = logging.getLogger(__name__)

class ChromaDBManager:
    """Manage ChromaDB vector database operations."""
    
    def __init__(self, collection_name: str = "human_rights_docs", 
                 persist_directory: Optional[str] = None):
        self.collection_name = collection_name
        self.persist_directory = persist_directory or Config.CHROMA_PERSIST_DIRECTORY
        
        # Ensure persist directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize embeddings
        self.embeddings = SentenceTransformerEmbeddings(Config.EMBEDDING_MODEL)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Initialize Langchain Chroma vectorstore
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        logger.info(f"ChromaDB initialized at {self.persist_directory}")
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector database."""
        try:
            # Add documents to vectorstore
            doc_ids = self.vectorstore.add_documents(documents)
            
            # Note: persist() is deprecated in newer versions, data is auto-persisted
            try:
                self.vectorstore.persist()
            except AttributeError:
                # persist() method doesn't exist in newer versions
                pass
            
            logger.info(f"Added {len(documents)} documents to ChromaDB")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
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
            logger.debug(f"ChromaDB search returned {len(results)} results")
            # Ensure results are always (Document, score) tuples
            fixed_results = []
            for item in results:
                if isinstance(item, tuple) and len(item) == 2:
                    doc, score = item
                elif isinstance(item, dict) and 'document' in item and 'score' in item:
                    doc = item['document']
                    score = item['score']
                else:
                    logger.warning(f"Unexpected result format in ChromaDBManager: {item}")
                    continue
                fixed_results.append((doc, float(score)))
            logger.debug(f"ChromaDB extracted {len(fixed_results)} document results")
            return fixed_results
        except Exception as e:
            logger.error(f"Error in similarity search with scores: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            collection = self.client.get_collection(self.collection_name)
            count = collection.count()
            
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory
            }
            
        except Exception as e:
            logger.warning(f"Could not get collection stats: {e}")
            return {
                "collection_name": self.collection_name,
                "document_count": 0,
                "persist_directory": self.persist_directory
            }
    
    def delete_collection(self):
        """Delete the entire collection."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise
    
    def update_document(self, doc_id: str, document: Document):
        """Update a specific document."""
        try:
            # Delete old document and add new one
            # Note: ChromaDB doesn't have direct update, so we delete and re-add
            self.vectorstore.delete([doc_id])
            new_id = self.vectorstore.add_documents([document])
            
            # Note: persist() is deprecated in newer versions, data is auto-persisted
            try:
                self.vectorstore.persist()
            except AttributeError:
                # persist() method doesn't exist in newer versions
                pass
            
            logger.info(f"Updated document {doc_id}")
            return new_id[0]
            
        except Exception as e:
            logger.error(f"Error updating document: {e}")
            raise
    
    def check_document_exists(self, file_path: str, file_hash: Optional[str] = None) -> Dict[str, Any]:
        """Check if a document already exists in the database."""
        try:
            collection = self.client.get_collection(self.collection_name)
            
            # Query by file path in metadata
            if file_path is not None:
                where_clause: Dict[str, Any] = {"source": file_path}
            else:
                where_clause: Dict[str, Any] = {}

            results = collection.get(
                where=where_clause,
                include=['metadatas', 'documents']
            )
            
            if results['ids']:
                # Document exists, return details
                return {
                    "exists": True,
                    "database": "ChromaDB",
                    "file_path": file_path,
                    "document_count": len(results['ids']),
                    "chunk_ids": results['ids']
                }
            
            # If file hash is provided, also check by hash
            if file_hash:
                # Ensure file_hash is a string or a type compatible with LiteralValue
                where_clause_hash: Dict[str, Any] = {"file_hash": str(file_hash)}
                results_hash = collection.get(
                    where=where_clause_hash,
                    include=['metadatas']
                )
                
                if results_hash['ids']:
                    return {
                        "exists": True,
                        "database": "ChromaDB",
                        "file_path": file_path,
                        "file_hash": file_hash,
                        "document_count": len(results_hash['ids']),
                        "chunk_ids": results_hash['ids'],
                        "detected_by": "file_hash"
                    }
            
            return {
                "exists": False,
                "database": "ChromaDB",
                "file_path": file_path
            }
            
        except Exception as e:
            logger.error(f"Error checking document existence in ChromaDB: {e}")
            return {
                "exists": False,
                "database": "ChromaDB",
                "file_path": file_path,
                "error": str(e)
            }
    
    def get_all_document_sources(self) -> List[str]:
        """Get all unique document source paths in the database."""
        try:
            collection = self.client.get_collection(self.collection_name)
            results = collection.get(include=['metadatas'])
            
            sources = set()
            metadatas = results.get('metadatas', [])
            if metadatas is None:
                metadatas = []
            for metadata in metadatas:
                if metadata and 'source' in metadata:
                    sources.add(metadata['source'])
            
            return list(sources)
            
        except Exception as e:
            logger.error(f"Error getting document sources from ChromaDB: {e}")
            return []
