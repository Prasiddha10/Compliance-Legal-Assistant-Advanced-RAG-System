"""Database management utilities."""
from typing import List, Dict, Any, Union
from langchain.schema import Document
from .chroma_db import ChromaDBManager
from .pinecone_db import PineconeDBManager
import logging

logger = logging.getLogger(__name__)

class DatabaseComparator:
    """Compare results between different vector databases."""
    
    def __init__(self):
        # Initialize ChromaDB
        try:
            self.chroma_db = ChromaDBManager()
            logger.info("ChromaDB initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize ChromaDB: {e}")
            self.chroma_db = None
        
        # Initialize Pinecone (only if API key is available)
        try:
            from src.config import Config
            if Config.PINECONE_API_KEY:
                self.pinecone_db = PineconeDBManager()
                logger.info("Pinecone initialized successfully")
            else:
                logger.warning("Pinecone API key not found - Pinecone will not be available")
                self.pinecone_db = None
        except Exception as e:
            logger.warning(f"Could not initialize Pinecone: {e}")
            self.pinecone_db = None
    
    def add_documents_to_all(self, documents: List[Document]) -> Dict[str, Any]:
        """Add documents to all available databases."""
        results = {}
        
        if self.chroma_db:
            try:
                chroma_ids = self.chroma_db.add_documents(documents)
                results["chroma"] = {
                    "success": True,
                    "doc_ids": chroma_ids,
                    "count": len(chroma_ids)
                }
            except Exception as e:
                results["chroma"] = {
                    "success": False,
                    "error": str(e)
                }
        
        if self.pinecone_db:
            try:
                pinecone_ids = self.pinecone_db.add_documents(documents)
                results["pinecone"] = {
                    "success": True,
                    "doc_ids": pinecone_ids,
                    "count": len(pinecone_ids)
                }
            except Exception as e:
                results["pinecone"] = {
                    "success": False,
                    "error": str(e)
                }
        
        return results
    
    def compare_search_results(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Compare search results from both databases."""
        results = {
            "query": query,
            "k": k,
            "chroma_results": [],
            "pinecone_results": [],
            "comparison_metrics": {}
        }
        
        # Search in ChromaDB
        if self.chroma_db:
            try:
                chroma_docs = self.chroma_db.similarity_search_with_score(query, k)
                results["chroma_results"] = [
                    {
                        "content": doc.page_content[:200] + "...",
                        "score": score,
                        "metadata": doc.metadata
                    }
                    for doc, score in chroma_docs
                ]
            except Exception as e:
                results["chroma_error"] = str(e)
        
        # Search in Pinecone
        if self.pinecone_db:
            try:
                pinecone_docs = self.pinecone_db.similarity_search_with_score(query, k)
                results["pinecone_results"] = [
                    {
                        "content": doc.page_content[:200] + "...",
                        "score": score,
                        "metadata": doc.metadata
                    }
                    for doc, score in pinecone_docs
                ]
            except Exception as e:
                results["pinecone_error"] = str(e)
        
        # Calculate comparison metrics
        results["comparison_metrics"] = self._calculate_comparison_metrics(
            results["chroma_results"], 
            results["pinecone_results"]
        )
        
        return results
    
    def _calculate_comparison_metrics(self, chroma_results: List[Dict], 
                                    pinecone_results: List[Dict]) -> Dict[str, Any]:
        """Calculate metrics comparing the two result sets."""
        metrics = {}
        
        if not chroma_results or not pinecone_results:
            return metrics
        
        # Extract content for comparison
        chroma_contents = [r["content"] for r in chroma_results]
        pinecone_contents = [r["content"] for r in pinecone_results]
        
        # Calculate overlap
        overlap = len(set(chroma_contents) & set(pinecone_contents))
        total_unique = len(set(chroma_contents) | set(pinecone_contents))
        
        metrics["overlap_ratio"] = overlap / len(chroma_results) if chroma_results else 0
        metrics["jaccard_similarity"] = overlap / total_unique if total_unique > 0 else 0
        
        # Average scores
        if chroma_results:
            metrics["chroma_avg_score"] = sum(r["score"] for r in chroma_results) / len(chroma_results)
        
        if pinecone_results:
            metrics["pinecone_avg_score"] = sum(r["score"] for r in pinecone_results) / len(pinecone_results)
        
        return metrics
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics from all databases."""
        stats = {}
        
        if self.chroma_db:
            try:
                stats["chroma"] = self.chroma_db.get_collection_stats()
            except Exception as e:
                stats["chroma"] = {"error": str(e)}
        
        if self.pinecone_db:
            try:
                stats["pinecone"] = self.pinecone_db.get_index_stats()
            except Exception as e:
                stats["pinecone"] = {"error": str(e)}
        
        return stats
    
    def compare_retrieval(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Compare retrieval results from both databases with timing."""
        import time
        results = {}
        
        # Search in ChromaDB
        if self.chroma_db:
            try:
                start_time = time.time()
                chroma_docs = self.chroma_db.similarity_search(query, k)
                chroma_time = time.time() - start_time
                
                results["chroma"] = {
                    "success": True,
                    "documents": chroma_docs,
                    "time_taken": chroma_time,
                    "count": len(chroma_docs)
                }
            except Exception as e:
                results["chroma"] = {
                    "success": False,
                    "error": str(e),
                    "documents": [],
                    "time_taken": 0,
                    "count": 0
                }
        
        # Search in Pinecone
        if self.pinecone_db:
            try:
                start_time = time.time()
                pinecone_docs = self.pinecone_db.similarity_search(query, k)
                pinecone_time = time.time() - start_time
                
                results["pinecone"] = {
                    "success": True,
                    "documents": pinecone_docs,
                    "time_taken": pinecone_time,
                    "count": len(pinecone_docs)
                }
            except Exception as e:
                results["pinecone"] = {
                    "success": False,
                    "error": str(e),
                    "documents": [],
                    "time_taken": 0,
                    "count": 0
                }
        
        return results
