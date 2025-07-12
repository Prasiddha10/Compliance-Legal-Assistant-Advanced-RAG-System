"""
Language utilities for multilingual support in the RAG system.
"""
import re
from typing import Dict, Optional, Tuple
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import logging

logger = logging.getLogger(__name__)

class LanguageProcessor:
    """Handles language detection and processing for multilingual queries."""
    
    SUPPORTED_LANGUAGES = {
        'en': 'English',
        'ne': 'Nepali',
        'auto': 'Auto-detect'
    }
    
    # Common Nepali words for better detection
    NEPALI_INDICATORS = [
        'नेपाल', 'संविधान', 'अधिकार', 'मानव', 'स्वतन्त्रता', 'समानता',
        'न्यायालय', 'सरकार', 'राज्य', 'कानून', 'नियम', 'धारा',
        'के', 'छ', 'गर्न', 'भन्न', 'मा', 'को', 'का', 'की'
    ]
    
    def __init__(self):
        """Initialize the language processor."""
        self.language_prompts = {
            'en': {
                'system_prompt': """You are a knowledgeable legal assistant specializing in human rights law. 
                Provide accurate, clear, and comprehensive answers based on the provided context. 
                Focus on human rights principles, legal precedents, and constitutional protections.
                Always cite relevant sources and maintain a professional, informative tone.""",
                'response_template': "Based on the human rights documents and legal context provided, here is the answer to your query:"
            },
            'ne': {
                'system_prompt': """तपाईं मानवअधिकार कानूनमा विशेषज्ञता प्राप्त एक जानकार कानूनी सहायक हुनुहुन्छ। 
                प्रदान गरिएको सन्दर्भको आधारमा सटीक, स्पष्ट र व्यापक जवाफ दिनुहोस्। 
                मानवअधिकारका सिद्धान्तहरू, कानूनी उदाहरणहरू र संवैधानिक सुरक्षाहरूमा केन्द्रित हुनुहोस्।
                सधैं सान्दर्भिक स्रोतहरू उद्धृत गर्नुहोस् र व्यावसायिक, जानकारीमूलक टोन कायम राख्नुहोस्।""",
                'response_template': "प्रदान गरिएको मानवअधिकार कागजातहरू र कानूनी सन्दर्भको आधारमा, यहाँ तपाईंको प्रश्नको जवाफ छ:"
            }
        }
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of input text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Language code ('en', 'ne', or 'auto' if uncertain)
        """
        try:
            # Remove special characters and numbers for better detection
            clean_text = re.sub(r'[^\w\s]', '', text)
            
            # Check for Nepali indicators first
            nepali_word_count = sum(1 for word in self.NEPALI_INDICATORS if word in text)
            if nepali_word_count > 0:
                return 'ne'
            
            # Use langdetect for other languages
            detected = detect(clean_text)
            
            # Map detected language to supported languages
            if detected in ['ne', 'hi']:  # Hindi might be detected as Nepali
                return 'ne'
            elif detected in ['en']:
                return 'en'
            else:
                # Default to English for unsupported languages
                return 'en'
                
        except (LangDetectException, Exception) as e:
            logger.warning(f"Language detection failed: {e}")
            return 'en'  # Default to English
    
    def get_language_prompt(self, language: str) -> Dict[str, str]:
        """
        Get language-specific prompts and templates.
        
        Args:
            language: Language code ('en', 'ne', or 'auto')
            
        Returns:
            Dictionary with system_prompt and response_template
        """
        if language == 'auto':
            return self.language_prompts['en']  # Default to English
        
        return self.language_prompts.get(language, self.language_prompts['en'])
    
    def process_query(self, query: str, language: str = 'auto') -> Tuple[str, Dict[str, str]]:
        """
        Process a query and determine the appropriate language handling.
        
        Args:
            query: User query
            language: Requested language ('auto', 'en', 'ne')
            
        Returns:
            Tuple of (detected_language, language_prompts)
        """
        if language == 'auto':
            detected_language = self.detect_language(query)
            logger.info(f"Auto-detected language: {detected_language} for query: {query[:50]}...")
        else:
            detected_language = language
            logger.info(f"Using specified language: {detected_language}")
        
        language_prompts = self.get_language_prompt(detected_language)
        
        return detected_language, language_prompts
    
    def format_response(self, response: str, language: str, sources: list) -> str:
        """
        Format the response according to the detected language.
        
        Args:
            response: Generated response
            language: Language code
            sources: List of source documents
            
        Returns:
            Formatted response
        """
        prompts = self.get_language_prompt(language)
        
        # Add language-specific formatting
        if language == 'ne':
            # Add Nepali-specific formatting
            formatted_response = f"{prompts['response_template']}\n\n{response}"
            
            # Add source citations in Nepali
            if sources:
                formatted_response += "\n\n**स्रोतहरू:**"
                for i, source in enumerate(sources, 1):
                    if source.get('metadata', {}).get('source'):
                        formatted_response += f"\n{i}. {source['metadata']['source']}"
        else:
            # English formatting
            formatted_response = f"{prompts['response_template']}\n\n{response}"
            
            # Add source citations in English
            if sources:
                formatted_response += "\n\n**Sources:**"
                for i, source in enumerate(sources, 1):
                    if source.get('metadata', {}).get('source'):
                        formatted_response += f"\n{i}. {source['metadata']['source']}"
        
        return formatted_response
    
    def get_language_stats(self) -> Dict[str, str]:
        """Get language support statistics."""
        return {
            'supported_languages': str(len(self.SUPPORTED_LANGUAGES)),
            'languages': ', '.join(self.SUPPORTED_LANGUAGES.keys()),
            'language_names': ', '.join(self.SUPPORTED_LANGUAGES.values())
        }
