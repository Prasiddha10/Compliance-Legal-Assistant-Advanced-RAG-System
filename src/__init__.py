"""Main package for the Human Rights RAG System."""
from .config import Config
from .rag import RAGPipeline
from .database import DatabaseComparator, ChromaDBManager, PineconeDBManager
from .evaluation import RAGEvaluationSuite
from .utils import PDFProcessor, SentenceTransformerEmbeddings

__version__ = "1.0.0"
__author__ = "RAG Development Team"
__description__ = "Human Rights Legal Assistant with RAG, LangGraph, and Comprehensive Evaluation"

__all__ = [
    "Config",
    "RAGPipeline", 
    "DatabaseComparator",
    "ChromaDBManager", 
    "PineconeDBManager",
    "RAGEvaluationSuite",
    "PDFProcessor",
    "SentenceTransformerEmbeddings"
]
