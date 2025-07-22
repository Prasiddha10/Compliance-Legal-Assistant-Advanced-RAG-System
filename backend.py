"""
FastAPI backend for the Compliance Legal Assistant RAG System
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import time
import os
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import RAG components
from src.rag.pipeline import RAGPipeline
from src.evaluation.rag_evaluator import RAGEvaluationSuite

# Global variables
rag_pipeline = None
evaluation_suite = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup the RAG system"""
    global rag_pipeline, evaluation_suite

    try:
        logger.info("Initializing RAG system...")

        # Check API keys first
        from src.config import Config
        missing_keys = []
        if not Config.OPENAI_API_KEY:
            missing_keys.append("OPENAI_API_KEY")
        if not Config.PINECONE_API_KEY:
            missing_keys.append("PINECONE_API_KEY")

        if missing_keys:
            logger.error(f"Missing required API keys: {', '.join(missing_keys)}")
            logger.error("Please create a .env file with your API keys. See .env.template for reference.")
            # Continue with limited functionality

        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline()
        logger.info("RAG Pipeline initialized")

        # Initialize evaluation suite
        evaluation_suite = RAGEvaluationSuite()
        logger.info("Evaluation Suite initialized")

        logger.info("RAG system initialized successfully!")
        yield

    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        logger.error("The system will run with limited functionality.")
        # Set minimal fallback instances
        rag_pipeline = None
        evaluation_suite = None
        yield
    finally:
        logger.info("Shutting down RAG system...")

app = FastAPI(
    title="Compliance Legal Assistant",
    description="RAG-powered legal assistant for compliance queries",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    language: Optional[str] = "auto"
    model: Optional[str] = "gpt-3.5-turbo"

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    database_used: str
    processing_time: float
    context_length: int
    model_used: str
    language_used: str
    detected_language: str
    evaluation_scores: Optional[Dict[str, Any]] = None
    web_search_results: Optional[List[Dict[str, Any]]] = None

class EvaluationRequest(BaseModel):
    queries: List[str]
    include_detailed_metrics: bool = True

class DocumentUploadResponse(BaseModel):
    filename: str
    status: str
    message: str
    file_hash: Optional[str] = None
    is_duplicate: bool = False
    existing_in_databases: Optional[List[Dict[str, Any]]] = None
    num_documents: int = 0
    file_size: Optional[int] = None

# Serve static files
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# Routes
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend HTML page."""
    html_path = Path("frontend/index.html")
    if html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(content="""
        <html>
            <head><title>Compliance Legal Assistant</title></head>
            <body>
                <h1>Compliance Legal Assistant</h1>
                <p>Frontend files not found. Please ensure frontend/index.html exists.</p>
            </body>
        </html>
        """)

@app.get("/{html_file}")
async def serve_html_files(html_file: str):
    """Serve HTML files from the frontend directory."""
    if not html_file.endswith('.html'):
        raise HTTPException(status_code=404, detail="File not found")
    
    html_path = Path(f"frontend/{html_file}")
    if html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    else:
        raise HTTPException(status_code=404, detail="File not found")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "system_initialized": rag_pipeline is not None,
        "timestamp": time.time()
    }

@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a user query and return the answer with sources."""
    if not rag_pipeline:
        # Return a helpful error message when RAG system is not initialized
        return QueryResponse(
            answer="I apologize, but the RAG system is not fully initialized. This could be due to:\n\n" +
                   "1. Missing API keys (OpenAI, Groq, Gemini)\n" +
                   "2. Database connection issues\n" +
                   "3. Configuration problems\n\n" +
                   "Please check the server logs for more details and ensure all API keys are properly configured.",
            sources=[],
            database_used="none",
            processing_time=0.0,
            context_length=0,
            model_used="none",
            language_used="none",
            detected_language="none",
            evaluation_scores=None
        )
    
    try:
        start_time = time.time()
        
        # Process query through RAG pipeline with specified model and language
        result = rag_pipeline.query(request.query, model_name=request.model or "gpt-3.5-turbo", language=request.language or "en")
        
        processing_time = time.time() - start_time
        
        # Format sources
        sources = []
        if result.get('retrieved_docs'):
            for doc in result['retrieved_docs']:
                sources.append({
                    "content": doc.get('content', ''),
                    "metadata": doc.get('metadata', {}),
                    "source": doc.get('metadata', {}).get('source', 'Unknown')
                })
        
        # Run evaluation if evaluation suite is available and response is valid
        evaluation_scores = None
        if evaluation_suite and not result.get('error'):
            try:
                response_text = result.get('response', '')
                # Only evaluate if we have a valid response (not an error message)
                if response_text and not response_text.startswith("Error generating response:"):
                    eval_result = evaluation_suite.evaluate_single_query(request.query, response_text)
                    evaluation_scores = {
                        "overall_score": eval_result.get('overall_performance', 0.0),
                        "generation_score": eval_result.get('generation_metrics', {}).get('overall_quality', 0.0),
                        "judge_score": eval_result.get('judge_evaluation', {}).get('comprehensive_score', 0.0) / 10.0 if eval_result.get('judge_evaluation', {}).get('comprehensive_score') else 0.0
                    }
                else:
                    # Set low scores for error responses
                    evaluation_scores = {
                        "overall_score": 0.0,
                        "generation_score": 0.0,
                        "judge_score": 0.0
                    }
            except Exception as e:
                logger.error(f"Error during evaluation: {e}")
                evaluation_scores = {
                    "overall_score": 0.0,
                    "generation_score": 0.0,
                    "judge_score": 0.0
                }
        
        # Extract web search results if available
        web_search_results = result.get('metadata', {}).get('web_search_results', [])

        return QueryResponse(
            answer=result.get('response', 'No response generated'),
            sources=sources,
            database_used=result.get('database_used', 'unknown'),
            processing_time=processing_time,
            context_length=len(result.get('response', '')),
            model_used=request.model or "gpt-3.5-turbo",
            language_used=request.language or "auto",
            detected_language=result.get('detected_language', 'en'),
            evaluation_scores=evaluation_scores,
            web_search_results=web_search_results
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return QueryResponse(
            answer=f"Sorry, I encountered an error while processing your query: {str(e)}\n\n" +
                   "Please try again or contact support if the issue persists.",
            sources=[],
            database_used="error",
            processing_time=0.0,
            context_length=0,
            model_used="error",
            language_used="error",
            detected_language="error",
            evaluation_scores=None,
            web_search_results=[]
        )

@app.post("/api/upload", response_model=DocumentUploadResponse)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload and process a document with duplicate detection."""
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    try:
        # Save uploaded file
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        file_path = upload_dir / file.filename
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process document immediately to check for duplicates
        if rag_pipeline and hasattr(rag_pipeline, 'add_documents'):
            result = rag_pipeline.add_documents(str(file_path))
            logger.info(f"Document processing result: {result}")
            
            # Check if document was a duplicate
            if result.get("status") == "duplicate":
                # Document already exists
                return DocumentUploadResponse(
                    filename=file.filename,
                    status="duplicate",
                    message=result.get("message", f"Document '{file.filename}' already exists in the vector databases"),
                    file_hash=result.get("file_hash"),
                    is_duplicate=True,
                    existing_in_databases=result.get("existing_in_databases", []),
                    num_documents=result.get("num_documents", 0),
                    file_size=result.get("file_size")
                )
            else:
                # Document was successfully processed
                return DocumentUploadResponse(
                    filename=file.filename,
                    status="success",
                    message=result.get("message", f"Document '{file.filename}' uploaded and processed successfully"),
                    file_hash=result.get("file_hash"),
                    is_duplicate=False,
                    existing_in_databases=None,
                    num_documents=result.get("num_documents", 0),
                    file_size=result.get("file_size")
                )
        else:
            logger.warning("add_documents method not available or RAG pipeline not initialized")
            return DocumentUploadResponse(
                filename=file.filename,
                status="error",
                message="RAG pipeline not properly initialized"
            )
        
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

@app.post("/api/evaluate")
async def run_evaluation(request: EvaluationRequest):
    """Run evaluation on a set of queries."""
    if not evaluation_suite:
        raise HTTPException(status_code=500, detail="Evaluation suite not initialized")
    
    try:
        evaluation_results = []
        total_scores = []
        
        for query in request.queries:
            ground_truth_docs = None  # No ground_truth_docs since queries are strings
            if rag_pipeline:
                # Generate response for the query
                result = rag_pipeline.query(query)
                response_text = result.get('response', '')
                
                # Run evaluation if we have a valid response
                if response_text and not response_text.startswith("Error generating response:"):
                    eval_result = evaluation_suite.evaluate_response(
                        query, response_text, result.get('context', ''),
                        ground_truth_docs=ground_truth_docs, rag_result=result
                    )
                    
                    overall_score = eval_result.get('overall_performance', 0.0)
                    generation_score = eval_result.get('generation_metrics', {}).get('overall_quality', 0.0)
                    judge_score_raw = eval_result.get('judge_evaluation', {}).get('comprehensive_score', 0.0)
                    judge_score = judge_score_raw / 10.0 if judge_score_raw else 0.0

                    # Extract retrieval score properly
                    retrieval_metrics = eval_result.get('retrieval_metrics', {})
                    retrieval_score = retrieval_metrics.get('avg_score', 0.0)

                    # Debug logging
                    logger.info(f"Evaluation API - Retrieval metrics: {retrieval_metrics}")
                    logger.info(f"Evaluation API - Initial retrieval score: {retrieval_score}")

                    # If avg_score is 0, try other retrieval metrics
                    if retrieval_score == 0.0:
                        retrieval_score = max(
                            retrieval_metrics.get('avg_relevance', 0.0),
                            retrieval_metrics.get('max_score', 0.0),
                            retrieval_metrics.get('ndcg@5', 0.0),
                            retrieval_metrics.get('ndcg@3', 0.0)
                        )
                        logger.info(f"Evaluation API - Fallback retrieval score: {retrieval_score}")

                    logger.info(f"Evaluation API - Final retrieval score for '{query[:30]}...': {retrieval_score}")

                    # Format evaluation result
                    formatted_result = {
                        "query": query,
                        "response": response_text,
                        "overall_score": overall_score,
                        "retrieval_score": retrieval_score,  # Add retrieval score to top level
                        "generation_score": generation_score,
                        "judge_score": judge_score,
                        "database_used": result.get('database_used', 'unknown'),
                        "processing_time": eval_result.get('processing_time', 0.0),
                        "llm_model": result.get('metadata', {}).get('model_used', 'unknown'),
                        "detailed_metrics": {
                            "retrieval_score": retrieval_score,
                            "generation_score": generation_score,
                            "judge_score": judge_score,
                            "processing_time": eval_result.get('processing_time', 0.0),
                            "retrieval_details": retrieval_metrics  # Include full retrieval metrics
                        } if request.include_detailed_metrics else None
                    }
                    evaluation_results.append(formatted_result)
                    total_scores.append(overall_score)
                else:
                    # Handle error responses
                    error_result = {
                        "query": query,
                        "response": response_text,
                        "overall_score": 0.0,
                        "retrieval_score": 0.0,
                        "generation_score": 0.0,
                        "judge_score": 0.0,
                        "error": "Failed to generate response"
                    }
                    evaluation_results.append(error_result)
                    total_scores.append(0.0)
            else:
                error_result = {
                    "query": query,
                    "response": "RAG pipeline not available",
                    "overall_score": 0.0,
                    "generation_score": 0.0,
                    "judge_score": 0.0,
                    "error": "RAG pipeline not initialized"
                }
                evaluation_results.append(error_result)
                total_scores.append(0.0)
        
        # Calculate summary statistics
        average_score = sum(total_scores) / len(total_scores) if total_scores else 0.0
        
        response_data = {
            "summary": {
                "total_queries": len(request.queries),
                "average_score": average_score,
                "timestamp": time.time()
            },
            "evaluation_results": evaluation_results
        }
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error running evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error running evaluation: {str(e)}")

@app.get("/api/stats")
async def get_stats():
    """Get system statistics."""
    try:
        # Check if RAG pipeline is initialized
        system_operational = rag_pipeline is not None and evaluation_suite is not None
        
        # Get actual database stats
        chroma_stats = {"status": "disconnected", "document_count": 0, "error": "Not initialized"}
        pinecone_stats = {"status": "disconnected", "document_count": 0, "error": "Not initialized"}
        
        if rag_pipeline and hasattr(rag_pipeline, 'db_comparator'):
            # Get ChromaDB stats
            try:
                if rag_pipeline.db_comparator.chroma_db:
                    chroma_collection_stats = rag_pipeline.db_comparator.chroma_db.get_collection_stats()
                    chroma_stats = {
                        "status": "connected",
                        "document_count": chroma_collection_stats.get("document_count", 0),
                        "error": None
                    }
                else:
                    chroma_stats["error"] = "ChromaDB not initialized"
            except Exception as e:
                chroma_stats = {
                    "status": "error",
                    "document_count": 0,
                    "error": str(e)
                }
            
            # Get Pinecone stats
            try:
                if rag_pipeline.db_comparator.pinecone_db:
                    pinecone_index_stats = rag_pipeline.db_comparator.pinecone_db.get_index_stats()
                    pinecone_stats = {
                        "status": "connected",
                        "document_count": pinecone_index_stats.get("total_vector_count", 0),
                        "error": None
                    }
                else:
                    pinecone_stats["error"] = "Pinecone not initialized"
            except Exception as e:
                pinecone_stats = {
                    "status": "error",
                    "document_count": 0,
                    "error": str(e)
                }
        
        stats = {
            "system_status": "operational" if system_operational else "offline",
            "system_initialized": rag_pipeline is not None,
            "evaluation_available": evaluation_suite is not None,
            "timestamp": time.time(),
            "databases": {
                "chroma": chroma_stats,
                "pinecone": pinecone_stats
            }
        }
        
        if rag_pipeline:
            stats.update({
                "models_available": 4,  # We know we have 4 models initialized
                "database_status": "connected"
            })
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return {
            "system_status": "error",
            "error": str(e),
            "databases": {
                "chroma": {
                    "status": "error",
                    "error": str(e)
                },
                "pinecone": {
                    "status": "error", 
                    "error": str(e)
                }
            }
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8005))
    uvicorn.run(app, host="0.0.0.0", port=port)
