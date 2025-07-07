"""Embedding utilities for vector representations."""
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
import logging

logger = logging.getLogger(__name__)

class SentenceTransformerEmbeddings(Embeddings):
    """Custom embeddings class using SentenceTransformers."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        logger.info(f"Loaded embedding model: {model_name}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return [list(map(float, emb.tolist() if hasattr(emb, 'tolist') else emb)) for emb in embeddings]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        embedding = self.model.encode([text], convert_to_tensor=False)
        return embedding[0].tolist()

class EmbeddingUtils:
    """Utilities for embedding operations."""
    
    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1_arr = np.array(vec1)
        vec2_arr = np.array(vec2)
        
        dot_product = np.dot(vec1_arr, vec2_arr)
        norm_vec1 = np.linalg.norm(vec1_arr)
        norm_vec2 = np.linalg.norm(vec2_arr)
        
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        
        return dot_product / (norm_vec1 * norm_vec2)
    
    @staticmethod
    def batch_embeddings(texts: List[str], model: SentenceTransformerEmbeddings, 
                        batch_size: int = 32) -> List[List[float]]:
        """Process embeddings in batches for large datasets."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = model.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
            
            if i % (batch_size * 10) == 0:
                logger.info(f"Processed {i + len(batch)}/{len(texts)} embeddings")
        
        return all_embeddings
