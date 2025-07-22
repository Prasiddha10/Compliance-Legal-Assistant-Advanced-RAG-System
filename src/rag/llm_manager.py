"""LLM clients for different providers."""
from typing import Dict, Any, List, Optional
import logging

# Core imports
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from src.config import Config

# Try to import OpenAI
try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    ChatOpenAI = None

# Try to import Groq
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None
logger = logging.getLogger(__name__)

class GroqLLM(LLM):
    """Custom LLM wrapper for Groq API."""
    
    model_name: str = "mixtral-8x7b-32768"
    max_tokens: int = 1024
    temperature: float = 0.1
    client: Any = None  # Add client as a field
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not GROQ_AVAILABLE or Groq is None:
            raise ImportError("Groq module not available. Install with: pip install groq")
        if not Config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is required")

        # Initialize Groq client with only the API key
        self.client = Groq(api_key=Config.GROQ_API_KEY)
        self.model_name = kwargs.get("model_name", Config.GROQ_MODEL)
        self.max_tokens = kwargs.get("max_tokens", 1024)
        self.temperature = kwargs.get("temperature", 0.1)
    
    @property
    def _llm_type(self) -> str:
        return "groq"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the Groq API."""
        try:
            if not self.client:
                raise ValueError("Groq client is not initialized")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=stop,
                **kwargs
            )
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error calling Groq API: {e}")
            raise

class LLMManager:
    """Manage different LLM providers."""
    
    def __init__(self):
        self.models = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize available LLM models."""
        # OpenAI models
        if Config.OPENAI_API_KEY and OPENAI_AVAILABLE and ChatOpenAI is not None:
            try:
                # Set the API key in environment for langchain-openai
                import os
                os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY

                self.models["gpt-3.5-turbo"] = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0.1
                )

                self.models["gpt-4"] = ChatOpenAI(
                    model="gpt-4",
                    temperature=0.1
                )
                logger.info("OpenAI models initialized")

            except Exception as e:
                logger.warning(f"Could not initialize OpenAI models: {e}")
        elif not OPENAI_AVAILABLE:
            logger.warning("OpenAI module not available. Install with: pip install langchain-openai")
        
        # Groq models
        if Config.GROQ_API_KEY and GROQ_AVAILABLE:
            try:
                # Only keep supported Groq models (remove gemma2-9b-it and llama3-8b-8192)
                self.models["llama-3.1-8b-instant"] = GroqLLM(
                    model_name="llama-3.1-8b-instant",
                    temperature=0.1,
                    max_tokens=1024
                )
                logger.info("Groq models initialized")
            except Exception as e:
                logger.warning(f"Could not initialize Groq models: {e}")
        elif not GROQ_AVAILABLE:
            logger.warning("Groq module not available. Install with: pip install groq")
    def get_model(self, model_name: Optional[str] = None):
        """Get a specific model or default model."""
        if model_name and model_name in self.models:
            return self.models[model_name]
        
        # Return first available model as default
        if self.models:
            default_model = list(self.models.keys())[0]
            logger.info(f"Using default model: {default_model}")
            return self.models[default_model]
        
        raise ValueError("No LLM models available")
    
    def list_available_models(self) -> List[str]:
        """List all available model names."""
        return list(self.models.keys())

    def generate_response(self, prompt: str, model_name: Optional[str] = None, **kwargs) -> str:
        """Generate a response using the specified model."""
        try:
            model = self.get_model(model_name)

            # Handle different model types
            if hasattr(model, 'invoke'):
                # LangChain ChatModel
                response = model.invoke(prompt, **kwargs)
                return response.content if hasattr(response, 'content') else str(response)
            elif hasattr(model, '__call__'):
                # Custom model with __call__ method
                return model(prompt, **kwargs)
            else:
                # Fallback for other model types
                return str(model.predict(prompt, **kwargs))

        except Exception as e:
            logger.error(f"Error generating response with model {model_name}: {e}")
            raise
    
    def test_model(self, model_name: str, test_prompt: str = "Hello, how are you?") -> Dict[str, Any]:
        """Test a specific model."""
        if model_name not in self.models:
            return {"error": f"Model {model_name} not available"}
        
        try:
            model = self.models[model_name]
            
            import time
            start_time = time.time()
            
            if hasattr(model, 'invoke'):
                response = model.invoke(test_prompt)
                response_text = response.content if hasattr(response, 'content') else str(response)
            else:
                response_text = model(test_prompt)
            
            end_time = time.time()
            
            return {
                "model": model_name,
                "test_prompt": test_prompt,
                "response": response_text,
                "response_time": end_time - start_time,
                "success": True
            }
            
        except Exception as e:
            return {
                "model": model_name,
                "test_prompt": test_prompt,
                "error": str(e),
                "success": False
            }
