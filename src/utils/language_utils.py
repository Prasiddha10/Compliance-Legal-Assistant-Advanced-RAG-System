"""
Language utilities for compliance-focused RAG system.
"""
import re
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class LanguageProcessor:
    """Handles language processing for compliance queries (English only)."""

    SUPPORTED_LANGUAGES = {
        'en': 'English',
        'ne': 'Nepali',
        'auto': 'Auto-detect'
    }

    # Common Nepali words for better detection
    NEPALI_INDICATORS = [
        # Core Nepali words
        'नेपाल', 'संविधान', 'अधिकार', 'मानव', 'स्वतन्त्रता', 'समानता',
        'न्यायालय', 'सरकार', 'राज्य', 'कानून', 'नियम', 'धारा',
        'के', 'छ', 'गर्न', 'भन्न', 'मा', 'को', 'का', 'की',
        # Compliance-related Nepali terms
        'अनुपालन', 'नियामक', 'दायित्व', 'आवश्यकता', 'मापदण्ड', 'निर्देशन',
        'कार्यान्वयन', 'पालना', 'व्यवस्था', 'प्रक्रिया', 'नीति', 'ऐन',
        'विनियम', 'निर्देशिका', 'मार्गदर्शन', 'सुरक्षा', 'गोपनीयता',
        # Question words in Nepali
        'के', 'कसरी', 'कहाँ', 'कहिले', 'किन', 'कुन', 'कति', 'कसले',
        # Common verbs and particles
        'हुन्छ', 'गर्छ', 'भन्छ', 'छैन', 'हुँदैन', 'गर्दैन', 'भएको', 'गरेको'
    ]

    def __init__(self):
        """Initialize the language processor with multilingual support."""
        self.language_prompts = {
            'en': {
                'system_prompt': """You are a knowledgeable compliance assistant specializing in regulatory frameworks and compliance requirements.
                Provide accurate, clear, and comprehensive answers based on the provided context.
                Focus on compliance requirements, regulatory standards, data protection, and legal obligations.
                Always cite relevant sources and maintain a professional, informative tone.
                When discussing compliance requirements, clearly distinguish between mandatory obligations and best practices.
                Do not generate any web search results, links, or "Additional Web Resources" sections.
                Only provide the main answer content.""",
                'response_template': "Based on the compliance documents and regulatory context provided, here is the answer to your query:"
            },
            'ne': {
                'system_prompt': """तपाईं नियामक ढाँचाहरू र अनुपालन आवश्यकताहरूमा विशेषज्ञता प्राप्त एक जानकार अनुपालन सहायक हुनुहुन्छ।

                निर्देशनहरू:
                - प्रदान गरिएको सन्दर्भको आधारमा सटीक, विस्तृत र व्यापक जवाफ दिनुहोस्
                - यदि सन्दर्भमा सान्दर्भिक कानूनी प्रावधानहरू छन् भने, विशिष्ट धारा वा खण्ड नम्बरहरू उद्धृत गर्नुहोस्
                - यदि सन्दर्भमा पर्याप्त जानकारी छैन भने, उपलब्ध जानकारी प्रदान गर्नुहोस् र के छुटेको छ स्पष्ट रूपमा भन्नुहोस्
                - अनुपालन आवश्यकताहरू, नियामक दायित्वहरू र कानूनी प्रक्रियाहरूमा केन्द्रित हुनुहोस्
                - कानूनी सटीकता कायम राख्दै स्पष्ट, पहुँचयोग्य भाषा प्रयोग गर्नुहोस्
                - अनुपालन आवश्यकताहरूको बारेमा छलफल गर्दा, अनिवार्य दायित्वहरू र उत्तम अभ्यासहरू बीच स्पष्ट रूपमा छुट्याउनुहोस्
                - सधैं नेपाली भाषामा जवाफ दिनुहोस्
                - व्यावसायिक र जानकारीमूलक टोन कायम राख्नुहोस्
                - कुनै पनि वेब खोजी परिणामहरू, लिङ्कहरू, वा "Additional Web Resources" खण्ड उत्पन्न नगर्नुहोस्
                - केवल मुख्य जवाफ मात्र प्रदान गर्नुहोस्""",
                'response_template': "प्रदान गरिएको अनुपालन कागजातहरू र नियामक सन्दर्भको आधारमा, यहाँ तपाईंको प्रश्नको जवाफ छ:",
                'web_search_intro': "कागजातहरूमा सान्दर्भिक जानकारी फेला नपरेकोले, यहाँ वेब खोजी परिणामहरू छन्:",
                'no_results': "माफ गर्नुहोस्, तपाईंको प्रश्नसँग सम्बन्धित जानकारी फेला परेन। कृपया अर्को प्रश्न सोध्नुहोस्।"
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
            import re
            clean_text = re.sub(r'[^\w\s]', '', text)

            # Check for Nepali indicators first
            nepali_word_count = sum(1 for word in self.NEPALI_INDICATORS if word in text)
            if nepali_word_count > 0:
                return 'ne'

            # Use langdetect for other languages
            try:
                from langdetect import detect
                detected = detect(clean_text)

                # Map detected language to supported languages
                if detected in ['ne', 'hi']:  # Hindi might be detected as Nepali
                    return 'ne'
                elif detected in ['en']:
                    return 'en'
                else:
                    # Default to English for unsupported languages
                    return 'en'
            except ImportError:
                # If langdetect is not available, default to English
                return 'en'

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
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
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Auto-detected language: {detected_language} for query: {query[:50]}...")
        else:
            detected_language = language
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Using specified language: {detected_language}")

        language_prompts = self.get_language_prompt(detected_language)

        return detected_language, language_prompts
    
    def format_response(self, response: str, language: str, sources: list = None, web_search_results: list = None) -> str:
        """
        Format the response for compliance system in the specified language.

        Args:
            response: Generated response
            language: Language code ('en' or 'ne')
            sources: List of source documents (optional)
            web_search_results: List of web search results (optional)

        Returns:
            Formatted response in the specified language
        """
        prompts = self.get_language_prompt(language)

        # Format response with appropriate template
        formatted_response = f"{prompts['response_template']}\n\n{response}"

        # TEMPORARILY DISABLE web search addition to debug duplication
        # The LLM is generating its own web search results, so we don't need to add them
        if False:  # Disabled for now
            pass

        # Add source citations (but user preference is to remove citations)
        # Commenting out source citations based on user memory preference
        # if sources:
        #     if language == 'ne':
        #         formatted_response += "\n\n**स्रोतहरू:**"
        #     else:
        #         formatted_response += "\n\n**Sources:**"
        #     for i, source in enumerate(sources, 1):
        #         if source.get('metadata', {}).get('source'):
        #             formatted_response += f"\n{i}. {source['metadata']['source']}"

        return formatted_response
    
    def get_language_stats(self) -> Dict[str, str]:
        """Get language support statistics."""
        return {
            'supported_languages': str(len(self.SUPPORTED_LANGUAGES)),
            'languages': ', '.join(self.SUPPORTED_LANGUAGES.keys()),
            'language_names': ', '.join(self.SUPPORTED_LANGUAGES.values())
        }
