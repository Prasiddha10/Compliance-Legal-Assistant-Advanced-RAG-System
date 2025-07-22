"""RAG package for retrieval-augmented generation."""
try:
    from .llm_manager import LLMManager, GroqLLM, GROQ_AVAILABLE
    __all__ = [
        "LLMManager",
        "GroqLLM",
        "RAGPipeline",
        "RAGState",
        "GROQ_AVAILABLE"
    ]
except ImportError:
    from .llm_manager import LLMManager, GROQ_AVAILABLE
    __all__ = [
        "LLMManager",
        "RAGPipeline",
        "RAGState",
        "GROQ_AVAILABLE"
    ]

from .pipeline import RAGPipeline, RAGState
