"""RAG pipeline implementation using LangGraph."""
from typing import Dict, Any, List, TypedDict, Optional, Union
import logging
import time
import requests
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.database.comparator import DatabaseComparator
from src.rag.llm_manager import LLMManager
from src.utils.document_processor import DocumentProcessor
from src.utils.language_utils import LanguageProcessor
from src.utils.web_search import WebSearchManager
import json

logger = logging.getLogger(__name__)

class RAGState(TypedDict):
    """State for the RAG workflow."""
    query: str
    retrieved_docs: List[Document]
    context: str
    response: str
    metadata: Dict[str, Any]
    database_used: str
    retrieval_scores: List[float]
    error: Optional[str]
    language: str
    detected_language: str

class RAGPipeline:
    """RAG pipeline using LangGraph for workflow orchestration."""
    
    def __init__(self, database_comparator: Optional[DatabaseComparator] = None, 
                 llm_manager: Optional[LLMManager] = None):
        self.db_comparator = database_comparator or DatabaseComparator()
        self.llm_manager = llm_manager or LLMManager()
        self.document_processor = DocumentProcessor()
        self.language_processor = LanguageProcessor()
        self.web_search_manager = WebSearchManager()
        
        # Log supported file types
        supported_extensions = self.document_processor.get_supported_extensions()
        logger.info(f"Document processor initialized with support for: {', '.join(supported_extensions)}")
        
        # Initialize the workflow graph
        self.workflow = self._create_workflow()
        
        # Initialize memory for conversation
        self.memory = MemorySaver()
        
        # Create the compiled graph (fast mode - no evaluation)
        self.app = self.workflow.compile(checkpointer=self.memory)
        
        logger.info("RAG Pipeline initialized with LangGraph and language processing")
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for RAG."""
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("rerank", self._rerank_documents)
        workflow.add_node("generate", self._generate_response)
        workflow.add_node("evaluate", self._evaluate_response)
        
        # Add edges - Route through evaluation for full workflow
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "generate")
        workflow.add_edge("generate", "evaluate")
        workflow.add_edge("evaluate", END)
        
        # Set entry point
        workflow.set_entry_point("retrieve")
        
        return workflow
    
    def _retrieve_documents(self, state: RAGState) -> RAGState:
        """Retrieve relevant documents from vector databases."""
        query = state["query"]
        
        try:
            # Compare results from both databases with reduced retrieval count for speed
            comparison_results = self.db_comparator.compare_search_results(query, k=7)  # Reduced from 15 to 7 for speed
            
            # Use ChromaDB results as primary, fallback to Pinecone
            if "chroma_results" in comparison_results and comparison_results["chroma_results"]:
                retrieved_docs = []
                scores = []
                
                # Apply compliance priority boost
                boosted_results = self._apply_compliance_priority(comparison_results["chroma_results"], query)
                
                for result in boosted_results[:5]:  # Take top 5 documents for faster processing
                    doc = Document(
                        page_content=result["content"],
                        metadata=result.get("metadata", {})
                    )
                    retrieved_docs.append(doc)
                    scores.append(result.get("score", 0.0))
                
                state["database_used"] = "chroma"
                
            elif "pinecone_results" in comparison_results and comparison_results["pinecone_results"]:
                retrieved_docs = []
                scores = []
                
                # Apply compliance priority boost
                boosted_results = self._apply_compliance_priority(comparison_results["pinecone_results"], query)
                
                for result in boosted_results[:5]:  # Take top 5 documents for faster processing
                    doc = Document(
                        page_content=result["content"],
                        metadata=result.get("metadata", {})
                    )
                    retrieved_docs.append(doc)
                    scores.append(result.get("score", 0.0))
                
                state["database_used"] = "pinecone"
            else:
                retrieved_docs = []
                scores = []
                state["database_used"] = "none"
            
            state["retrieved_docs"] = retrieved_docs
            state["retrieval_scores"] = scores

            # Check if we should trigger web search based on relevance threshold
            should_use_web_search = self._should_trigger_web_search(retrieved_docs, scores, query)
            state["use_web_search"] = should_use_web_search
            
            # Ensure metadata is properly initialized
            if "metadata" not in state or state["metadata"] is None:
                state["metadata"] = {}
            
            state["metadata"].update({
                "retrieval_results": comparison_results,
                "num_docs_retrieved": len(retrieved_docs)
            })
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents using {state['database_used']}")
            
        except Exception as e:
            logger.error(f"Error in document retrieval: {e}")
            state["error"] = str(e)
            state["retrieved_docs"] = []
            state["retrieval_scores"] = []
            if "metadata" not in state:
                state["metadata"] = {}
        
        return state
    
    def _apply_compliance_priority(self, results, query):
        """Apply enhanced priority boost to compliance and regulatory documents for compliance queries."""

        # Enhanced compliance keyword categories with different boost levels
        compliance_categories = {
            'high_priority': {
                'keywords': ['gdpr', 'compliance', 'regulation', 'requirement', 'obligation', 'mandatory'],
                'boost_factor': 0.4  # Strongest boost
            },
            'medium_priority': {
                'keywords': ['standard', 'framework', 'procedure', 'guideline', 'policy', 'directive'],
                'boost_factor': 0.6
            },
            'domain_specific': {
                'keywords': ['data protection', 'privacy', 'audit', 'assessment', 'monitoring',
                           'certification', 'cybersecurity', 'risk management', 'ai systems'],
                'boost_factor': 0.5
            },
            'legal_terms': {
                'keywords': ['article', 'section', 'provision', 'statute', 'law', 'legal requirement'],
                'boost_factor': 0.7
            }
        }

        # Determine query compliance level and type
        query_lower = query.lower()
        query_compliance_score = 0.0
        query_categories = []

        for category, config in compliance_categories.items():
            category_matches = sum(1 for keyword in config['keywords'] if keyword in query_lower)
            if category_matches > 0:
                query_compliance_score += category_matches * (1.0 - config['boost_factor'])
                query_categories.append(category)

        # If not a compliance query, return original results
        if query_compliance_score == 0:
            return results

        # Enhanced document classification and boosting
        enhanced_results = []

        for result in results:
            source = result.get("metadata", {}).get("source", "").lower()
            content = result.get("content", "").lower()
            metadata = result.get("metadata", {})

            # Calculate document compliance relevance
            doc_compliance_score = 0.0
            doc_boost_factor = 1.0  # Default (no boost)

            # Check content relevance by category
            for category, config in compliance_categories.items():
                if category in query_categories:  # Only boost for relevant categories
                    content_matches = sum(1 for keyword in config['keywords']
                                        if keyword in content or keyword in source)
                    if content_matches > 0:
                        doc_compliance_score += content_matches
                        # Use the strongest boost factor for this document
                        doc_boost_factor = min(doc_boost_factor, config['boost_factor'])

            # Additional metadata-based boosting
            if metadata:
                doc_type = metadata.get('document_type', '').lower()
                regulation = metadata.get('regulation', '').lower()
                topic = metadata.get('topic', '').lower()

                if doc_type in ['regulation', 'framework', 'requirements', 'standard']:
                    doc_boost_factor = min(doc_boost_factor, 0.5)
                    doc_compliance_score += 2

                if regulation in ['gdpr', 'eu ai act', 'basel iii'] and any(reg in query_lower for reg in ['gdpr', 'ai', 'financial']):
                    doc_boost_factor = min(doc_boost_factor, 0.3)
                    doc_compliance_score += 3

                if topic and any(topic_word in query_lower for topic_word in topic.split()):
                    doc_boost_factor = min(doc_boost_factor, 0.6)
                    doc_compliance_score += 1

            # Apply boosting
            original_score = result.get("score", 0.0)
            if doc_compliance_score > 0:
                # Apply boost (lower score is better in most vector DBs)
                boosted_score = original_score * doc_boost_factor
                result["score"] = boosted_score
                result["boosted"] = True
                result["compliance_score"] = doc_compliance_score
                result["boost_factor"] = doc_boost_factor

            enhanced_results.append(result)

        # Sort by boosted scores and compliance relevance
        enhanced_results.sort(key=lambda x: (x.get("score", 1.0), -x.get("compliance_score", 0)))

        return enhanced_results

    def _should_trigger_web_search(self, retrieved_docs: List[Document], scores: List[float], query: str) -> bool:
        """Determine if web search should be triggered based on document relevance."""
        # No documents found - definitely use web search
        if not retrieved_docs or not scores:
            return True

        # Check average relevance score (lower scores mean less relevant in most vector DBs)
        avg_score = sum(scores) / len(scores) if scores else 0

        # For ChromaDB/Pinecone, lower scores typically mean higher similarity
        # But we want to check if the scores are too high (meaning low relevance)
        relevance_threshold = 1.5  # Adjust based on your vector DB scoring

        # If average score is too high (low relevance), trigger web search
        if avg_score > relevance_threshold:
            logger.info(f"Low relevance detected (avg score: {avg_score:.3f}), triggering web search")
            return True

        # Check if query contains terms not likely to be in compliance documents
        non_compliance_indicators = [
            'cryptocurrency', 'crypto', 'bitcoin', 'blockchain', 'nft', 'defi',
            'sports', 'entertainment', 'celebrity', 'movie', 'music', 'game',
            'weather', 'news', 'current events', '2024', '2025', 'latest', 'recent',
            'breaking', 'update', 'new', 'fresh'
        ]

        query_lower = query.lower()
        non_compliance_matches = sum(1 for term in non_compliance_indicators if term in query_lower)

        # If query has many non-compliance terms, consider web search
        if non_compliance_matches >= 2:
            logger.info(f"Non-compliance query detected ({non_compliance_matches} indicators), triggering web search")
            return True

        # Check document content relevance
        if retrieved_docs:
            # Sample first document content
            first_doc_content = retrieved_docs[0].page_content.lower() if retrieved_docs[0].page_content else ""

            # Check if query terms appear in the document
            query_terms = query_lower.split()
            content_matches = sum(1 for term in query_terms if term in first_doc_content)

            # If very few query terms match document content, consider web search
            if len(query_terms) > 2 and content_matches < len(query_terms) * 0.3:
                logger.info(f"Low content relevance detected ({content_matches}/{len(query_terms)} terms match), triggering web search")
                return True

        return False
    
    def _rerank_documents(self, state: RAGState) -> RAGState:
        """Rerank documents based on relevance to query."""
        try:
            retrieved_docs = state["retrieved_docs"]
            scores = state["retrieval_scores"]
            
            if retrieved_docs and scores and len(retrieved_docs) == len(scores):
                # Sort by score (assuming higher scores are better)
                sorted_pairs = sorted(zip(retrieved_docs, scores), 
                                    key=lambda x: x[1], reverse=True)
                
                state["retrieved_docs"] = [doc for doc, _ in sorted_pairs]
                state["retrieval_scores"] = [score for _, score in sorted_pairs]
                
                logger.info(f"Reranked {len(retrieved_docs)} documents")
            else:
                logger.warning("Could not rerank documents: mismatch in docs and scores")
            
        except Exception as e:
            logger.error(f"Error in document reranking: {e}")
            state["error"] = str(e)
        
        return state

    def _web_search(self, query: str, num_results: int = 2) -> List[Dict[str, str]]:
        """Perform web search when no relevant documents are found."""
        try:
            # Use the web search manager for real web search
            web_results = self.web_search_manager.search(query, num_results)

            logger.info(f"Web search returned {len(web_results)} results for query: {query[:50]}...")
            return web_results

        except Exception as e:
            logger.error(f"Error in web search: {e}")
            # Fallback to mock search if web search fails
            return self.web_search_manager._mock_search(query, num_results)

    def _clean_response(self, response: str) -> str:
        """Clean response to remove repetition and limit length."""
        try:
            # Split into sentences
            sentences = response.split('।')  # Nepali sentence delimiter
            if len(sentences) == 1:
                sentences = response.split('.')  # English sentence delimiter

            # Remove excessive repetition
            cleaned_sentences = []
            seen_sentences = set()

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                # Check for repetition (allow max 1 similar sentence - be more aggressive)
                similar_count = sum(1 for seen in seen_sentences if self._sentences_similar(sentence, seen))
                if similar_count < 1:
                    cleaned_sentences.append(sentence)
                    seen_sentences.add(sentence)

                # Limit total sentences to prevent extremely long responses
                if len(cleaned_sentences) >= 10:
                    break

            # Rejoin sentences and handle incomplete ones
            if '।' in response:
                # Filter out very short incomplete sentences
                cleaned_sentences = [s for s in cleaned_sentences if len(s.strip()) > 10]
                cleaned_response = '।'.join(cleaned_sentences)
                if cleaned_response and not cleaned_response.endswith('।'):
                    cleaned_response += '।'
            else:
                # Filter out very short incomplete sentences
                cleaned_sentences = [s for s in cleaned_sentences if len(s.strip()) > 10]
                cleaned_response = '. '.join(cleaned_sentences)
                if cleaned_response and not cleaned_response.endswith('.'):
                    cleaned_response += '.'

            return cleaned_response

        except Exception as e:
            logger.error(f"Error cleaning response: {e}")
            # Fallback: just truncate if cleaning fails
            return response[:2000] + "..." if len(response) > 2000 else response

    def _sentences_similar(self, sent1: str, sent2: str) -> bool:
        """Check if two sentences are similar (for repetition detection)."""
        # Simple similarity check based on common words
        words1 = set(sent1.lower().split())
        words2 = set(sent2.lower().split())

        if not words1 or not words2:
            return False

        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        similarity = intersection / union if union > 0 else 0
        return similarity > 0.5  # 50% similarity threshold (more aggressive)

    def _remove_llm_web_sections(self, response: str) -> str:
        """Remove any web search sections that the LLM might have generated."""
        try:
            logger.info(f"Cleaning response of length: {len(response)}")

            # More aggressive approach - split by common web search markers
            # and only keep the part before any web search content

            # List of markers that indicate start of web search content
            web_markers = [
                'Additional Web Resources:',
                'additional web resources:',
                'Web Search Results:',
                'web search results:',
                'Google Search:',
                'google search:',
                'Bing Search:',
                'bing search:',
                'Search for \'',
                'search for \'',
                'Find the latest',
                'find the latest',
                'Discover comprehensive',
                'discover comprehensive'
            ]

            # Find the earliest occurrence of any web marker
            earliest_pos = len(response)
            for marker in web_markers:
                pos = response.find(marker)
                if pos != -1 and pos < earliest_pos:
                    earliest_pos = pos

            # If we found a web marker, cut the response there
            if earliest_pos < len(response):
                cleaned_response = response[:earliest_pos].strip()
                logger.info(f"Removed web section starting at position {earliest_pos}")
                logger.info(f"Cleaned response length: {len(cleaned_response)}")
                return cleaned_response

            logger.info("No web sections found to remove")
            return response

        except Exception as e:
            logger.error(f"Error removing LLM web sections: {e}")
            return response

    def _generate_response(self, state: RAGState) -> RAGState:
        """Generate response using LLM with retrieved context and language support."""
        try:
            retrieved_docs = state["retrieved_docs"]
            query = state["query"]
            model_name = state.get("metadata", {}).get("model_name", "gpt-3.5-turbo")
            detected_language = state.get("detected_language", "en")
            language_prompts = state.get("metadata", {}).get("language_prompts", {})
            
            # Create context from retrieved documents or trigger web search
            web_search_results = []
            use_web_search = state.get("use_web_search", False)

            # Debug logging
            logger.info(f"Generation step - use_web_search flag: {use_web_search}")

            # Fallback: Re-check if we should use web search (in case flag wasn't passed)
            if not use_web_search and retrieved_docs:
                scores = state.get("retrieval_scores", [])
                use_web_search = self._should_trigger_web_search(retrieved_docs, scores, query)
                logger.info(f"Generation step - re-evaluated web search need: {use_web_search}")

            if use_web_search:
                # Trigger web search for better results
                logger.info(f"Web search triggered for query: {query[:50]}...")
                web_search_results = self._web_search(query)
                logger.info(f"Web search returned {len(web_search_results)} results")

                # When web search is triggered, provide minimal context to prevent detailed answers
                # The LLM should not generate detailed responses for non-database queries
                context = "The database does not contain sufficient information to answer this specific query. Please refer to the web search results for current information."

                # Store web search results in metadata
                if "metadata" not in state:
                    state["metadata"] = {}
                state["metadata"]["web_search_results"] = web_search_results

                logger.info(f"Web search triggered for query: {query[:50]}... Found {len(web_search_results)} results")
                logger.info(f"Web search results stored in metadata: {len(web_search_results)} items")

                # Debug: Log first result
                if web_search_results:
                    logger.info(f"First web search result: {web_search_results[0]}")

            elif retrieved_docs:
                # Use full document context when web search is not needed
                context_parts = []
                for i, doc in enumerate(retrieved_docs[:3]):  # Use top 3 documents
                    content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                    context_parts.append(f"Document {i+1}:\n{content}")

                context = "\n\n".join(context_parts)
            else:
                # Fallback case
                context = "No relevant information found."

            state["context"] = context
            
            # Use language-specific prompt if available
            if language_prompts and 'system_prompt' in language_prompts:
                system_prompt = language_prompts['system_prompt']
            else:
                system_prompt = """You are an expert legal assistant specializing in compliance law, regulatory frameworks, and international legal standards. You provide accurate, detailed guidance on compliance requirements, regulatory obligations, and legal procedures."""
            
            # Create the compliance-focused prompt template
            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template=f"""{system_prompt}

Use the following context from compliance and regulatory documents to answer the question.

Instructions:
- Provide accurate, detailed responses based on the provided context
- If the context contains relevant legal provisions, cite the specific article or section numbers when possible
- If the context doesn't contain enough information, provide what information is available and clearly state what is missing
- Focus on compliance requirements, regulatory obligations, and legal procedures
- Use clear, accessible language while maintaining legal accuracy
- When discussing compliance requirements, clearly distinguish between mandatory obligations and best practices

Context:
{{context}}

Question: {{question}}

Answer:"""
                )
            
            # Get LLM with specified model and generate response
            llm = self.llm_manager.get_model(model_name)

            # Set response limits for better performance
            if hasattr(llm, 'max_tokens'):
                llm.max_tokens = 800  # Increased for complete responses
            elif hasattr(llm, 'model_kwargs'):
                llm.model_kwargs = llm.model_kwargs or {}
                llm.model_kwargs['max_tokens'] = 800
            
            # Generate response based on LLM type
            try:
                if hasattr(llm, 'invoke'):
                    # For newer LangChain ChatModels
                    prompt = prompt_template.format(context=context, question=query)
                    response = llm.invoke(prompt)
                    response_text = response.content if hasattr(response, 'content') else str(response)
                elif callable(llm):
                    # For direct callable LLM
                    prompt = prompt_template.format(context=context, question=query)
                    response_text = llm(prompt)
                else:
                    # Fallback
                    response_text = "Error: Could not generate response - LLM not available"
                    state["error"] = "LLM not available"
                
                # Check for repetition and clean response
                response_text = self._clean_response(str(response_text))

                # Remove any web search sections generated by LLM
                response_text = self._remove_llm_web_sections(response_text)

                # Check if response contains an error
                if "Error generating response:" in str(response_text) or "Error code:" in str(response_text):
                    state["error"] = str(response_text)
                    response_text = f"Sorry, I encountered an error while generating the response. The model '{model_name}' may be temporarily unavailable. Please try a different model."
                    
            except Exception as e:
                logger.error(f"Error generating response with {model_name}: {e}")
                state["error"] = str(e)
                response_text = f"Error generating response: {str(e)}"
            
            # Format response with language-specific formatting
            web_search_results = state.get("metadata", {}).get("web_search_results", [])
            formatted_response = self.language_processor.format_response(
                str(response_text), detected_language, [], web_search_results
            )
            
            state["response"] = formatted_response
            
            # Update metadata
            if "metadata" not in state or state["metadata"] is None:
                state["metadata"] = {}
            
            state["metadata"].update({
                "context_length": len(context),
                "num_docs_used": len(retrieved_docs[:3]),
                "model_used": model_name,
                "web_search_results": web_search_results
            })
            
            logger.info("Generated response using LLM")
            
        except Exception as e:
            logger.error(f"Error in response generation: {e}")
            state["error"] = str(e)
            state["response"] = f"Error generating response: {str(e)}"
        
        return state
    
    def _evaluate_response(self, state: RAGState) -> RAGState:
        """Evaluate the generated response quality."""
        try:
            response = state["response"]
            context = state["context"]
            query = state["query"]
            
            # Simple evaluation metrics
            evaluation_metrics = {}
            
            # Response length
            evaluation_metrics["response_length"] = len(response)
            
            # Context utilization (check if response uses context)
            if context and "No relevant documents found" not in context:
                # Simple check for context utilization
                context_words = set(context.lower().split())
                response_words = set(response.lower().split())
                overlap = len(context_words & response_words)
                evaluation_metrics["context_utilization"] = overlap / len(context_words) if context_words else 0
            else:
                evaluation_metrics["context_utilization"] = 0
            
            # Query relevance (simple keyword matching)
            query_words = set(query.lower().split())
            response_words = set(response.lower().split())
            query_overlap = len(query_words & response_words)
            evaluation_metrics["query_relevance"] = query_overlap / len(query_words) if query_words else 0
            
            # Update metadata
            if "metadata" not in state:
                state["metadata"] = {}
            
            state["metadata"]["evaluation_metrics"] = evaluation_metrics
            
            logger.info("Evaluated response quality")
            
        except Exception as e:
            logger.error(f"Error in response evaluation: {e}")
            if "metadata" not in state:
                state["metadata"] = {}
            state["metadata"]["evaluation_error"] = str(e)
        
        return state
    
    def query(self, question: str, thread_id: str = "default", model_name: str = "gpt-3.5-turbo", language: str = "auto") -> Dict[str, Any]:
        """Process a query through the RAG pipeline with language support."""
        try:
            # Process language detection
            detected_language, language_prompts = self.language_processor.process_query(question, language)
            
            # Initial state with proper type checking
            initial_state: RAGState = {
                "query": question,
                "retrieved_docs": [],
                "context": "",
                "response": "",
                "metadata": {
                    "model_name": model_name,
                    "language": language,
                    "detected_language": detected_language,
                    "language_prompts": language_prompts
                },
                "database_used": "",
                "retrieval_scores": [],
                "error": None,
                "language": language,
                "detected_language": detected_language
            }
            
            # Run the workflow with proper config
            config = {"configurable": {"thread_id": thread_id}}
            final_state = self.app.invoke(initial_state, config=config)  # type: ignore[arg-type]
            
            # Ensure final_state has all required keys
            final_state = dict(final_state)  # Convert to regular dict if needed
            
            # Safe access to retrieved_docs
            retrieved_docs = final_state.get("retrieved_docs", [])
            retrieval_scores = final_state.get("retrieval_scores", [])
            formatted_docs = []
            # Show only top 3 docs in frontend, but pass all scores for evaluation
            for i, doc in enumerate(retrieved_docs):
                try:
                    if hasattr(doc, 'page_content'):
                        content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                    else:
                        content = str(doc)[:200] + "..."
                        metadata = {}
                    doc_dict = {
                        "content": content,
                        "metadata": metadata
                    }
                    if i < len(retrieval_scores):
                        doc_dict["score"] = retrieval_scores[i]
                    formatted_docs.append(doc_dict)
                except Exception as e:
                    logger.warning(f"Error formatting document: {e}")
                    continue
            # Only show top 3 docs in frontend, but pass all scores for evaluation
            return {
                "query": final_state.get("query", question),
                "response": final_state.get("response", "No response generated"),
                "retrieved_docs": formatted_docs[:3],
                "database_used": final_state.get("database_used", "none"),
                "retrieval_scores": retrieval_scores,
                "detected_language": detected_language,  # Add detected language to top level
                "metadata": final_state.get("metadata", {}),
                "error": final_state.get("error")
            }
            
        except Exception as e:
            logger.error(f"Error in RAG query processing: {e}")
            return {
                "query": question,
                "response": f"Error processing query: {str(e)}",
                "retrieved_docs": [],
                "database_used": "none",
                "retrieval_scores": [],
                "metadata": {"error_details": str(e)},
                "error": str(e)
            }
    
    def add_documents(self, file_path: str, metadata: Optional[Dict[str, Any]] = None, progress_callback=None) -> Dict[str, Any]:
        """Add documents to the vector databases with optional progress tracking and duplicate detection."""
        try:
            # Check if file type is supported
            if not self.document_processor.is_supported_file(file_path):
                supported_extensions = self.document_processor.get_supported_extensions()
                raise ValueError(f"Unsupported file type. Supported formats: {', '.join(supported_extensions)}")
            
            # Calculate file hash for duplicate detection
            if progress_callback:
                progress_callback("Checking for duplicates...", 0.1)
            
            file_hash = self.document_processor.calculate_file_hash(file_path)
            logger.info(f"Calculated file hash for {Path(file_path).name}: {file_hash}")
            
            # Check if document already exists in any database
            duplicate_check = self.db_comparator.check_document_exists_in_all(file_path, file_hash)
            
            if duplicate_check.get("exists_anywhere", False):
                # Document already exists, return detailed information
                existing_databases = []
                for db_name, db_result in duplicate_check.get("databases", {}).items():
                    if db_result.get("exists", False):
                        existing_databases.append({
                            "database": db_result.get("database", db_name),
                            "document_count": db_result.get("document_count", 0),
                            "detected_by": db_result.get("detected_by", "source_path")
                        })
                
                if progress_callback:
                    progress_callback("Document already exists", 1.0)
                
                return {
                    "file_path": file_path,
                    "file_hash": file_hash,
                    "status": "duplicate",
                    "message": f"Document '{Path(file_path).name}' already exists in the vector databases",
                    "existing_in_databases": existing_databases,
                    "num_documents": 0,
                    "skipped": True
                }
            
            # Process document if it's not a duplicate
            if progress_callback:
                progress_callback("Processing document...", 0.3)
            
            documents = self.document_processor.process_document(file_path, metadata)
            
            # Add file hash to metadata for all documents
            for doc in documents:
                doc.metadata["file_hash"] = file_hash
                doc.metadata["file_size"] = Path(file_path).stat().st_size
                doc.metadata["upload_timestamp"] = time.time()
            
            # Add to databases
            if progress_callback:
                progress_callback("Adding to databases...", 0.9)
            
            results = self.db_comparator.add_documents_to_all(documents)
            
            if progress_callback:
                progress_callback("Complete", 1.0)
            
            return {
                "file_path": file_path,
                "file_hash": file_hash,
                "status": "success",
                "message": f"Document '{Path(file_path).name}' successfully processed and added",
                "num_documents": len(documents),
                "database_results": results,
                "file_type": Path(file_path).suffix.lower(),
                "file_size": Path(file_path).stat().st_size
            }
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            if progress_callback:
                progress_callback(f"Error: {str(e)}", 0.0)
            return {
                "file_path": file_path,
                "num_documents": 0,
                "error": str(e)
            }
