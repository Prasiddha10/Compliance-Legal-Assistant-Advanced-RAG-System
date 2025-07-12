"""Configuration settings for the RAG system."""
import os
import warnings
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Suppress common warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
warnings.filterwarnings("ignore", category=UserWarning, module="pdfminer")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Set PyTorch to avoid loading registered classes warnings
os.environ.setdefault("PYTORCH_DISABLE_JIT_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

class Config:
    """Configuration class for the RAG system."""
    
    # API Keys  
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # LangSmith Configuration
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
    LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "rag-human-rights-evaluation")
    
    # Database Configuration
    CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")
    CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "human_rights_docs")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "human-rights-index")
    
    # Model Configuration  
    # Important: Match embedding dimension with Pinecone index dimension (1024)
    # Option 1: Use original model and update Pinecone index to 384 dims
    # Option 2: Use model that produces 1024 dims (current approach)
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))  # Actual dimension of all-MiniLM-L6-v2
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")
    
    # Processing Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "300"))
    
    # Evaluation Configuration
    EVAL_DATASET_SIZE = int(os.getenv("EVAL_DATASET_SIZE", "100"))
    EVAL_BATCH_SIZE = int(os.getenv("EVAL_BATCH_SIZE", "10"))
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    UPLOADS_DIR = BASE_DIR / "uploads"
    
    @classmethod
    def validate_config(cls, component=None):
        """Validate configuration for specific components or all."""
        
        if component == "openai" or component is None:
            if not cls.OPENAI_API_KEY:
                if component == "openai":
                    raise ValueError("OpenAI API key is required for this functionality")
                    
        if component == "groq" or component is None:
            if not cls.GROQ_API_KEY:
                if component == "groq":
                    raise ValueError("Groq API key is required for this functionality")
                    
        if component == "pinecone":
            if not cls.PINECONE_API_KEY:
                raise ValueError("Pinecone API key is required for Pinecone functionality")
                
        if component == "langsmith":
            if not cls.LANGCHAIN_API_KEY:
                raise ValueError("LangChain API key is required for LangSmith functionality")
        
        # Create required directories
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.UPLOADS_DIR.mkdir(exist_ok=True)
        Path(cls.CHROMA_PERSIST_DIRECTORY).mkdir(parents=True, exist_ok=True)
        
        return True
    
    @classmethod
    def get_available_components(cls):
        """Get list of available components based on configuration."""
        components = []
        
        if cls.OPENAI_API_KEY:
            components.append("openai")
        if cls.GROQ_API_KEY:
            components.append("groq")
        if cls.PINECONE_API_KEY:
            components.append("pinecone")
        if cls.LANGCHAIN_API_KEY:
            components.append("langsmith")
            
        components.append("chromadb")  # Always available
        components.append("pdf_processing")  # Always available
        
        return components
        return True
