"""RAG pipeline implementation using LangGraph."""
from typing import Dict, Any, List, TypedDict, Optional, Union
import logging
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.database.comparator import DatabaseComparator
from src.rag.llm_manager import LLMManager
from src.utils.pdf_processor import PDFProcessor
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

class RAGPipeline:
    """RAG pipeline using LangGraph for workflow orchestration."""
    
    def __init__(self, database_comparator: Optional[DatabaseComparator] = None, 
                 llm_manager: Optional[LLMManager] = None):
        self.db_comparator = database_comparator or DatabaseComparator()
        self.llm_manager = llm_manager or LLMManager()
        self.pdf_processor = PDFProcessor()
        
        # Initialize the workflow graph
        self.workflow = self._create_workflow()
        
        # Initialize memory for conversation
        self.memory = MemorySaver()
        
        # Create the compiled graph
        self.app = self.workflow.compile(checkpointer=self.memory)
        
        logger.info("RAG Pipeline initialized with LangGraph")
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for RAG."""
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("rerank", self._rerank_documents)
        workflow.add_node("generate", self._generate_response)
        workflow.add_node("evaluate", self._evaluate_response)
        
        # Add edges
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
            # Compare results from both databases
            comparison_results = self.db_comparator.compare_search_results(query, k=5)
            
            # Use ChromaDB results as primary, fallback to Pinecone
            if "chroma_results" in comparison_results and comparison_results["chroma_results"]:
                retrieved_docs = []
                scores = []
                
                for result in comparison_results["chroma_results"]:
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
                
                for result in comparison_results["pinecone_results"]:
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
    
    def _generate_response(self, state: RAGState) -> RAGState:
        """Generate response using LLM with retrieved context."""
        try:
            retrieved_docs = state["retrieved_docs"]
            query = state["query"]
            
            # Create context from retrieved documents
            if retrieved_docs:
                context_parts = []
                for i, doc in enumerate(retrieved_docs[:3]):  # Use top 3 documents
                    content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                    context_parts.append(f"Document {i+1}:\n{content}")
                
                context = "\n\n".join(context_parts)
            else:
                context = "No relevant documents found."
            
            state["context"] = context
            
            # Create the prompt
            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template="""You are an expert legal assistant specializing in human rights law. 
                Use the following context from human rights documents to answer the question.
                Provide accurate, detailed responses based only on the provided context.
                If the context doesn't contain enough information, clearly state that.
                
                Context:
                {context}
                
                Question: {question}
                
                Answer:"""
            )
            
            # Get LLM and generate response
            llm = self.llm_manager.get_model()
            
            # Generate response based on LLM type
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
            
            state["response"] = str(response_text)
            
            # Update metadata
            if "metadata" not in state or state["metadata"] is None:
                state["metadata"] = {}
            
            state["metadata"].update({
                "context_length": len(context),
                "num_docs_used": len(retrieved_docs[:3]),
                "model_used": self.llm_manager.list_available_models()[0] if self.llm_manager.list_available_models() else "unknown"
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
    
    def query(self, question: str, thread_id: str = "default") -> Dict[str, Any]:
        """Process a query through the RAG pipeline."""
        try:
            # Initial state with proper type checking
            initial_state: RAGState = {
                "query": question,
                "retrieved_docs": [],
                "context": "",
                "response": "",
                "metadata": {},
                "database_used": "",
                "retrieval_scores": [],
                "error": None
            }
            
            # Run the workflow with proper config
            config = {"configurable": {"thread_id": thread_id}}
            final_state = self.app.invoke(initial_state, config=config)  # type: ignore[arg-type]
            
            # Ensure final_state has all required keys
            final_state = dict(final_state)  # Convert to regular dict if needed
            
            # Safe access to retrieved_docs
            retrieved_docs = final_state.get("retrieved_docs", [])
            formatted_docs = []
            
            for doc in retrieved_docs[:3]:
                try:
                    if hasattr(doc, 'page_content'):
                        content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                    else:
                        content = str(doc)[:200] + "..."
                        metadata = {}
                    
                    formatted_docs.append({
                        "content": content,
                        "metadata": metadata
                    })
                except Exception as e:
                    logger.warning(f"Error formatting document: {e}")
                    continue
            
            # Format response with safe key access
            return {
                "query": final_state.get("query", question),
                "response": final_state.get("response", "No response generated"),
                "retrieved_docs": formatted_docs,
                "database_used": final_state.get("database_used", "none"),
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
                "metadata": {"error_details": str(e)},
                "error": str(e)
            }
    
    def add_documents(self, file_path: str, metadata: Optional[Dict[str, Any]] = None, progress_callback=None) -> Dict[str, Any]:
        """Add documents to the vector databases with optional progress tracking."""
        try:
            # Process PDF with progress callback
            documents = self.pdf_processor.process_pdf(file_path, metadata)
            
            # Add to databases
            if progress_callback:
                progress_callback("Adding to databases...", 0.95)
            
            results = self.db_comparator.add_documents_to_all(documents)
            
            if progress_callback:
                progress_callback("Complete", 1.0)
            
            return {
                "file_path": file_path,
                "num_documents": len(documents),
                "database_results": results
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
