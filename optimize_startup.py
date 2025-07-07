"""Optimization utilities for faster Streamlit app loading."""
import os
import logging
from pathlib import Path

def ensure_directories():
    """Ensure all required directories exist."""
    directories = [
        "uploads",
        "data",
        "chroma_db",
        "reports"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        
def warm_up_embeddings():
    """Pre-load embedding model to avoid cold start."""
    try:
        from src.utils.embeddings import SentenceTransformerEmbeddings
        from src.config import Config
        
        # Initialize embeddings once
        embeddings = SentenceTransformerEmbeddings(Config.EMBEDDING_MODEL)
        
        # Test with a simple sentence to warm up the model
        test_embedding = embeddings.embed_query("test")
        
        print(f"✅ Embedding model warmed up - dimension: {len(test_embedding)}")
        return embeddings
        
    except Exception as e:
        print(f"❌ Failed to warm up embeddings: {e}")
        return None

def optimize_for_streamlit():
    """Run optimization steps for Streamlit app."""
    print("🚀 Optimizing Streamlit app startup...")
    
    # Ensure directories exist
    ensure_directories()
    print("✅ Directories created")
    
    # Warm up embeddings
    embeddings = warm_up_embeddings()
    
    # Set logging level to reduce startup noise
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    
    print("✅ Optimization complete!")
    
    return {
        "embeddings_ready": embeddings is not None,
        "directories_ready": True
    }

if __name__ == "__main__":
    optimize_for_streamlit()
