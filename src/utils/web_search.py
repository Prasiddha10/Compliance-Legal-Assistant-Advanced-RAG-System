"""
Web Search Utility for Compliance Legal Assistant
Provides web search fallback when no relevant documents are found in the database.
"""

import logging
import requests
import time
from typing import List, Dict, Optional, Any
from urllib.parse import quote_plus
import json

logger = logging.getLogger(__name__)

class WebSearchManager:
    """Manages web search functionality with multiple search providers."""

    def __init__(self):
        """Initialize web search manager."""
        self.search_providers = {
            'google_bing': self._google_bing_search,
            'serper': self._serper_search,
            'mock': self._mock_search
        }
        self.default_provider = 'google_bing'  # Google and Bing search URLs

    def search(self, query: str, num_results: int = 2, provider: str = None) -> List[Dict[str, str]]:
        """
        Perform web search using specified provider.

        Args:
            query: Search query
            num_results: Number of results to return
            provider: Search provider to use ('google_bing', 'serper', 'mock')

        Returns:
            List of search results with title, url, and snippet
        """
        provider = provider or self.default_provider

        if provider not in self.search_providers:
            logger.warning(f"Unknown search provider: {provider}. Using default: {self.default_provider}")
            provider = self.default_provider

        try:
            # Add compliance-focused terms to improve results
            enhanced_query = self._enhance_compliance_query(query)

            logger.info(f"Performing web search with {provider} for: {enhanced_query[:100]}...")

            search_function = self.search_providers[provider]
            results = search_function(enhanced_query, num_results)

            # Filter and clean results
            cleaned_results = self._clean_search_results(results, query)

            logger.info(f"Web search returned {len(cleaned_results)} results")
            return cleaned_results

        except Exception as e:
            logger.error(f"Web search failed with {provider}: {e}")
            # Fallback to mock search
            if provider != 'mock':
                return self._mock_search(query, num_results)
            return []

    def _enhance_compliance_query(self, query: str) -> str:
        """Enhance query with compliance-focused terms in appropriate language."""
        # Detect if query is in Nepali
        nepali_indicators = ['नेपाल', 'कानून', 'नियम', 'अनुपालन', 'नियामक', 'के', 'छ', 'गर्न', 'मा', 'को', 'का', 'की']
        is_nepali = any(indicator in query for indicator in nepali_indicators)

        if is_nepali:
            # Nepali compliance terms
            nepali_compliance_terms = [
                "अनुपालन", "नियामक", "कानूनी आवश्यकताहरू", "नियामक ढाँचा",
                "कानूनी अनुपालन", "नियामक निर्देशन", "अनुपालन मापदण्डहरू"
            ]

            # If query doesn't contain Nepali compliance terms, add relevant ones
            if not any(term in query for term in nepali_compliance_terms):
                if any(word in query for word in ["डेटा", "गोपनीयता", "सुरक्षा"]):
                    query += " अनुपालन नियम"
                elif any(word in query for word in ["एआई", "कृत्रिम बुद्धिमत्ता"]):
                    query += " एआई नियामक अनुपालन"
                elif any(word in query for word in ["साइबर", "सुरक्षा"]):
                    query += " साइबर सुरक्षा अनुपालन"
                else:
                    query += " कानूनी अनुपालन"
        else:
            # English compliance terms
            compliance_terms = [
                "compliance", "regulation", "legal requirements", "regulatory framework",
                "legal compliance", "regulatory guidance", "compliance standards"
            ]

            query_lower = query.lower()

            # If query doesn't contain compliance terms, add relevant ones
            if not any(term in query_lower for term in compliance_terms):
                if any(word in query_lower for word in ["gdpr", "data protection", "privacy"]):
                    query += " compliance regulation"
                elif any(word in query_lower for word in ["ai", "artificial intelligence"]):
                    query += " AI regulation compliance"
                elif any(word in query_lower for word in ["cyber", "security"]):
                    query += " cybersecurity compliance"
                else:
                    query += " legal compliance"

        return query

    def _google_bing_search(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """Generate Google and Bing search URLs (no API required)."""
        try:
            # Create search URLs for Google and Bing
            search_engines = [
                {
                    'name': 'Google',
                    'url': f"https://www.google.com/search?q={quote_plus(query)}",
                    'title': f"Google Search: {query[:50]}{'...' if len(query) > 50 else ''}",
                    'snippet': f"Search for '{query}' on Google to find the latest compliance information, regulatory updates, and authoritative sources."
                },
                {
                    'name': 'Bing',
                    'url': f"https://www.bing.com/search?q={quote_plus(query)}",
                    'title': f"Bing Search: {query[:50]}{'...' if len(query) > 50 else ''}",
                    'snippet': f"Search for '{query}' on Bing to discover comprehensive compliance resources, legal guidance, and regulatory documentation."
                }
            ]

            # Always return exactly 2 results: Google and Bing only
            results = []
            for engine in search_engines[:2]:  # Only take first 2 (Google and Bing)
                results.append({
                    'title': engine['title'],
                    'url': engine['url'],
                    'snippet': engine['snippet']
                })

            logger.info(f"Generated {len(results)} Google/Bing search results")
            return results

        except Exception as e:
            logger.error(f"Google/Bing search generation failed: {e}")
            return self._mock_search(query, num_results)

    def _simple_web_search(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """Simple web search using Google and Bing only."""
        try:
            # Create search URLs for Google and Bing only
            search_engines = [
                {
                    'name': 'Google',
                    'url': f"https://www.google.com/search?q={quote_plus(query)}",
                    'title': f"Google Search: {query[:50]}{'...' if len(query) > 50 else ''}",
                    'snippet': f"Search for '{query}' on Google to find relevant compliance information and regulatory guidance."
                },
                {
                    'name': 'Bing',
                    'url': f"https://www.bing.com/search?q={quote_plus(query)}",
                    'title': f"Bing Search: {query[:50]}{'...' if len(query) > 50 else ''}",
                    'snippet': f"Search for '{query}' on Bing to find comprehensive compliance resources and legal documentation."
                }
            ]

            results = []
            for engine in search_engines[:min(num_results, 2)]:
                results.append({
                    'title': engine['title'],
                    'url': engine['url'],
                    'snippet': engine['snippet']
                })

            return results

        except Exception as e:
            logger.error(f"Simple web search failed: {e}")
            return []

    def _serper_search(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """Search using Serper API (requires API key)."""
        try:
            import os
            api_key = os.getenv('SERPER_API_KEY')

            if not api_key:
                logger.warning("SERPER_API_KEY not found, falling back to Google/Bing")
                return self._google_bing_search(query, num_results)

            url = "https://google.serper.dev/search"
            headers = {
                'X-API-KEY': api_key,
                'Content-Type': 'application/json'
            }

            payload = {
                'q': query,
                'num': num_results
            }

            response = requests.post(url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()

            data = response.json()
            results = []

            for item in data.get('organic', [])[:num_results]:
                results.append({
                    'title': item.get('title', '')[:100],
                    'url': item.get('link', ''),
                    'snippet': item.get('snippet', '')[:300]
                })

            return results

        except Exception as e:
            logger.error(f"Serper search failed: {e}")
            return self._google_bing_search(query, num_results)

    def _mock_search(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """Generate mock compliance-focused search results."""
        mock_results = [
            {
                "title": f"Compliance Requirements: {query[:50]}...",
                "url": "https://www.compliance-regulations.gov/requirements",
                "snippet": "Official compliance requirements and regulatory obligations for legal frameworks and industry standards. Find detailed guidance on regulatory compliance, legal obligations, and industry-specific requirements."
            },
            {
                "title": f"Regulatory Guidance: {query[:50]}...",
                "url": "https://www.regulatory-guidance.org/compliance",
                "snippet": "Comprehensive regulatory guidance and compliance best practices for organizations. Access detailed information on compliance frameworks, regulatory standards, and implementation guidelines."
            },
            {
                "title": f"Legal Compliance Framework: {query[:50]}...",
                "url": "https://www.legal-compliance.gov/frameworks",
                "snippet": "Search legal compliance frameworks, regulatory requirements, and industry-specific obligations. Find authoritative guidance on compliance implementation and regulatory adherence."
            },
            {
                "title": f"Industry Standards: {query[:50]}...",
                "url": "https://www.industry-standards.org/compliance",
                "snippet": "Industry-specific compliance standards and regulatory requirements. Access comprehensive information on sector-specific obligations and compliance best practices."
            },
            {
                "title": f"Compliance Documentation: {query[:50]}...",
                "url": "https://www.compliance-docs.gov/resources",
                "snippet": "Official compliance documentation, regulatory texts, and legal requirements. Find authoritative sources for compliance information and regulatory guidance."
            }
        ]

        # Limit to maximum 2 results
        return mock_results[:min(num_results, 2)]

    def _clean_search_results(self, results: List[Dict[str, str]], original_query: str) -> List[Dict[str, str]]:
        """Clean and validate search results."""
        cleaned_results = []

        for result in results:
            # Ensure required fields exist
            if not all(key in result for key in ['title', 'url', 'snippet']):
                continue

            # Clean and truncate fields
            cleaned_result = {
                'title': result['title'].strip()[:150],
                'url': result['url'].strip(),
                'snippet': result['snippet'].strip()[:400]
            }

            # Skip empty results
            if not cleaned_result['title'] or not cleaned_result['url']:
                continue

            # Add query relevance indicator
            cleaned_result['query'] = original_query
            cleaned_result['search_timestamp'] = time.time()

            cleaned_results.append(cleaned_result)

        return cleaned_results