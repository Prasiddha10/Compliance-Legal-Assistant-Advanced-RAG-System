"""LLM clients for different providers."""
from typing import Dict, Any, List, Optional
import openai
from groq import Groq
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_openai import ChatOpenAI
from src.config import Config
import logging

# Try to import Google Generative AI
try:
    import google.generativeai as genai
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False

logger = logging.getLogger(__name__)

class GroqLLM(LLM):
    """Custom LLM wrapper for Groq API."""
    
    model_name: str = "mixtral-8x7b-32768"
    max_tokens: int = 1024
    temperature: float = 0.1
    client: Any = None  # Add client as a field
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not Config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is required")
        
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
        if Config.OPENAI_API_KEY:
            try:
                from pydantic import SecretStr

                self.models["gpt-3.5-turbo"] = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0.1,
                    api_key=SecretStr(Config.OPENAI_API_KEY)
                )
                
                self.models["gpt-4"] = ChatOpenAI(
                    model="gpt-4",
                    temperature=0.1,
                    api_key=SecretStr(Config.OPENAI_API_KEY)
                )
                logger.info("OpenAI models initialized")
                
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI models: {e}")
        
        # Groq models
        if Config.GROQ_API_KEY:
            try:
                # Using currently available Groq models
                self.models["llama-3.1-8b-instant"] = GroqLLM(
                    model_name="llama-3.1-8b-instant",
                    temperature=0.1,
                    max_tokens=1024
                )
                
                # Add other available models
                self.models["mixtral-8x7b-32768"] = GroqLLM(
                    model_name="mixtral-8x7b-32768",
                    temperature=0.1,
                    max_tokens=1024
                )
                
                self.models["gemma2-9b-it"] = GroqLLM(
                    model_name="gemma2-9b-it",
                    temperature=0.1,
                    max_tokens=1024
                )
                
                logger.info("Groq models initialized")
                
            except Exception as e:
                logger.warning(f"Could not initialize Groq models: {e}")
        
        # Google Gemini models
        if GOOGLE_AI_AVAILABLE and Config.GOOGLE_API_KEY:
            try:
                # Use the current supported Gemini model
                from pydantic import SecretStr

                self.models["gemini-1.5-flash"] = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    temperature=0.1,
                    api_key=SecretStr(Config.GOOGLE_API_KEY)
                )
                # Also add the pro version if available
                self.models["gemini-1.5-pro"] = ChatGoogleGenerativeAI(
                    model="gemini-1.5-pro",
                    temperature=0.1,
                    api_key=SecretStr(Config.GOOGLE_API_KEY)
                )
                logger.info("Google Gemini models initialized")
                
            except Exception as e:
                logger.warning(f"Could not initialize Google Gemini models: {e}")
        
        if not self.models:
            logger.warning("No LLM models initialized. Please check your API keys.")
        else:
            logger.info(f"Initialized {len(self.models)} LLM models: {list(self.models.keys())}")
    
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
