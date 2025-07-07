"""Main Streamlit application for the RAG system."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
from pathlib import Path
import os

# Import our modules
from src.rag.pipeline import RAGPipeline
from src.database.comparator import DatabaseComparator
from src.evaluation.rag_evaluator import RAGEvaluationSuite
from src.utils.pdf_processor import PDFProcessor
from src.config import Config

# Page configuration
st.set_page_config(
    page_title="Human Rights RAG System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'evaluation_suite' not in st.session_state:
    st.session_state.evaluation_suite = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = []

def initialize_system():
    """Initialize the RAG system."""
    try:
        # Validate configuration
        Config.validate_config()
        
        # Initialize components
        st.session_state.rag_pipeline = RAGPipeline()
        st.session_state.evaluation_suite = RAGEvaluationSuite(st.session_state.rag_pipeline)
        
        return True
    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        return False

def main():
    """Main application function."""
    
    # Title and description
    st.title("⚖️ Human Rights Legal Assistant")
    st.markdown("*Powered by RAG with LangGraph, monitored by LangSmith, and evaluated comprehensively*")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Home", "📄 Document Upload", "💬 Chat Interface", "📊 Database Comparison", "🔬 Evaluation Dashboard", "⚙️ Settings"]
    )
    
    # System status
    with st.sidebar:
        st.subheader("System Status")
        
        # Check if system is initialized
        if st.session_state.rag_pipeline is None:
            st.warning("System not initialized")
            if st.button("Initialize System"):
                with st.spinner("Initializing RAG system..."):
                    if initialize_system():
                        st.success("System initialized successfully!")
                        st.rerun()
        else:
            st.success("System ready")
            
            # Performance monitoring
            st.subheader("Performance")
            if st.button("🧹 Clear PDF Cache"):
                from src.utils.pdf_processor import PDFProcessor
                pdf_processor = PDFProcessor()
                pdf_processor.clear_cache()
                st.success("Cache cleared!")
                st.rerun()
            
            # Show cache stats
            try:
                from src.utils.pdf_processor import PDFProcessor
                pdf_processor = PDFProcessor()
                cache_stats = pdf_processor.get_cache_stats()
                st.caption(f"PDF Cache: {cache_stats['text_cache_size']} texts, {cache_stats['document_cache_size']} doc sets")
            except Exception as e:
                st.caption("Cache stats unavailable")
            
            # System actions
            st.subheader("System Actions")
            
            # Database status
            try:
                db_stats = st.session_state.rag_pipeline.db_comparator.get_database_stats()
                
                if "chroma" in db_stats:
                    st.info(f"ChromaDB: {db_stats['chroma'].get('document_count', 0)} docs")
                
                if "pinecone" in db_stats:
                    st.info(f"Pinecone: {db_stats['pinecone'].get('total_vector_count', 0)} vectors")
                    
            except Exception as e:
                st.warning(f"Database status unavailable: {str(e)}")
    
    # Page routing
    if page == "🏠 Home":
        home_page()
    elif page == "📄 Document Upload":
        document_upload_page()
    elif page == "💬 Chat Interface":
        chat_interface_page()
    elif page == "📊 Database Comparison":
        database_comparison_page()
    elif page == "🔬 Evaluation Dashboard":
        evaluation_dashboard_page()
    elif page == "⚙️ Settings":
        settings_page()

def home_page():
    """Home page with system overview."""
    
    st.header("Welcome to the Human Rights Legal Assistant")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Features")
        st.markdown("""
        - **Document Processing**: Upload and process human rights PDFs
        - **Intelligent Search**: Query legal documents with natural language
        - **Multi-Database**: Compare ChromaDB vs Pinecone performance
        - **LangGraph Workflow**: Advanced RAG pipeline orchestration
        - **LangSmith Monitoring**: Real-time tracking and debugging
        - **Comprehensive Evaluation**: Retrieval, generation, and LLM-as-judge metrics
        - **Database Comparison**: Side-by-side vector database analysis
        """)
        
    with col2:
        st.subheader("🏗️ Architecture")
        st.markdown("""
        - **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
        - **Vector Stores**: ChromaDB + Pinecone
        - **LLMs**: OpenAI GPT + Groq (Mixtral, Llama2)
        - **Workflow**: LangGraph state machines
        - **Monitoring**: LangSmith tracing
        - **Evaluation**: RAGAS + Custom metrics + LLM Judge
        """)
    
    # Quick stats if system is initialized
    if st.session_state.rag_pipeline:
        st.subheader("📈 Quick Stats")
        
        col1, col2, col3, col4 = st.columns(4)
        
        try:
            db_stats = st.session_state.rag_pipeline.db_comparator.get_database_stats()
            
            with col1:
                chroma_docs = db_stats.get('chroma', {}).get('document_count', 0)
                st.metric("ChromaDB Documents", chroma_docs)
            
            with col2:
                pinecone_vectors = db_stats.get('pinecone', {}).get('total_vector_count', 0)
                st.metric("Pinecone Vectors", pinecone_vectors)
            
            with col3:
                chat_count = len(st.session_state.chat_history)
                st.metric("Queries Processed", chat_count)
            
            with col4:
                eval_count = len(st.session_state.evaluation_results)
                st.metric("Evaluations Run", eval_count)
                
        except Exception as e:
            st.warning(f"Could not load stats: {str(e)}")

def document_upload_page():
    """Document upload and processing page."""
    
    st.header("📄 Document Upload & Processing")
    
    if not st.session_state.rag_pipeline:
        st.warning("Please initialize the system first from the sidebar.")
        return
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload Human Rights Documents (PDF)",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload PDF documents related to human rights law, conventions, declarations, etc."
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.subheader(f"Processing: {uploaded_file.name}")
            
            # Save uploaded file
            upload_path = Path("uploads") / uploaded_file.name
            upload_path.parent.mkdir(exist_ok=True)
            
            with open(upload_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Create progress bar and status
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def progress_callback(step: str, percent: float):
                progress_bar.progress(percent)
                status_text.text(f"{step} ({percent*100:.0f}%)")
            
            # Process document with progress tracking
            try:
                # Add metadata
                metadata = {
                    "uploaded_by": "streamlit_user",
                    "upload_time": time.time(),
                    "document_category": "human_rights_law"
                }
                
                # Process and add to databases
                start_time = time.time()
                result = st.session_state.rag_pipeline.add_documents(
                    str(upload_path), metadata, progress_callback=progress_callback
                )
                processing_time = time.time() - start_time
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.success(f"✅ Processed {result['num_documents']} chunks in {processing_time:.2f}s")
                    
                    # Show processing metrics
                    if "database_results" in result:
                        st.json(result["database_results"])
                        
                    # Show cache stats
                    pdf_processor = PDFProcessor()
                    cache_stats = pdf_processor.get_cache_stats()
                    st.caption(f"Cache: {cache_stats['text_cache_size']} texts, {cache_stats['document_cache_size']} doc sets")
                    
                    # Show file size info
                    file_size_mb = uploaded_file.size / (1024 * 1024)
                    st.caption(f"File size: {file_size_mb:.2f} MB")
                
                with col2:
                    # Preview document content using optimized preview
                    with st.spinner("Loading preview..."):
                        pdf_processor = PDFProcessor()
                        preview_text = pdf_processor.get_text_preview(str(upload_path), max_chars=500)
                        
                        st.text_area(
                            "Document Preview",
                            preview_text,
                            height=200
                        )
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            
            # Clean up uploaded file
            try:
                upload_path.unlink()
            except:
                pass

def chat_interface_page():
    """Chat interface for querying the RAG system."""
    
    st.header("💬 Legal Assistant Chat")
    
    if not st.session_state.rag_pipeline:
        st.warning("Please initialize the system first from the sidebar.")
        return
    
    # Chat history display
    chat_container = st.container()
    
    # Query input
    with st.form("query_form"):
        user_query = st.text_area(
            "Ask a question about human rights law:",
            placeholder="e.g., What are the fundamental principles of the Universal Declaration of Human Rights?",
            height=100
        )
        
        col1, col2, col3 = st.columns([1, 1, 4])
        
        with col1:
            submit_query = st.form_submit_button("🔍 Ask", use_container_width=True)
        
        with col2:
            evaluate_response = st.checkbox("📊 Evaluate Response", value=True)
    
    # Process query
    if submit_query and user_query.strip():
        with st.spinner("Searching and generating response..."):
            try:
                # Get RAG response
                response = st.session_state.rag_pipeline.query(user_query)
                
                # Add to chat history
                chat_entry = {
                    "timestamp": time.time(),
                    "query": user_query,
                    "response": response,
                    "evaluation": None
                }
                
                # Run evaluation if requested
                if evaluate_response and st.session_state.evaluation_suite:
                    evaluation = st.session_state.evaluation_suite.evaluate_single_query(user_query)
                    chat_entry["evaluation"] = evaluation
                    st.session_state.evaluation_results.append(evaluation)
                
                st.session_state.chat_history.append(chat_entry)
                
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
    
    # Display chat history
    with chat_container:
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Q: {chat['query'][:50]}...", expanded=(i == 0)):
                
                # Query
                st.markdown(f"**Question:** {chat['query']}")
                
                # Response
                st.markdown(f"**Answer:** {chat['response'].get('response', 'No response generated')}")
                
                # Metadata
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Retrieved Documents:**")
                    for j, doc in enumerate(chat['response'].get('retrieved_docs', [])):
                        # Handle both Document objects and formatted dictionaries
                        if hasattr(doc, 'page_content'):
                            content = doc.page_content[:100]
                        elif isinstance(doc, dict) and 'content' in doc:
                            content = doc['content'][:100]
                        else:
                            content = str(doc)[:100]
                        st.text(f"{j+1}. {content}...")
                
                with col2:
                    st.markdown("**Metadata:**")
                    metadata = chat['response'].get('metadata', {})
                    st.json({
                        "database_used": chat['response'].get('database_used', 'unknown'),
                        "processing_time": f"{metadata.get('processing_time', 0):.2f}s",
                        "context_length": metadata.get('context_length', 0),
                        "model_used": metadata.get('model_used', 'unknown')
                    })
                
                # Evaluation results
                if chat.get('evaluation'):
                    st.markdown("**Evaluation Scores:**")
                    eval_data = chat['evaluation']
                    
                    score_cols = st.columns(4)
                    with score_cols[0]:
                        st.metric("Overall", f"{eval_data.get('overall_performance', 0):.3f}")
                    with score_cols[1]:
                        gen_quality = eval_data.get('generation_metrics', {}).get('overall_quality', 0)
                        st.metric("Generation", f"{gen_quality:.3f}")
                    with score_cols[2]:
                        judge_score = eval_data.get('judge_evaluation', {}).get('comprehensive_score', 0)
                        st.metric("Judge Score", f"{judge_score:.2f}/10")
                    with score_cols[3]:
                        processing_time = eval_data.get('processing_time', 0)
                        st.metric("Time", f"{processing_time:.2f}s")

def database_comparison_page():
    """Database comparison and analysis page."""
    
    st.header("📊 Database Comparison")
    
    if not st.session_state.rag_pipeline:
        st.warning("Please initialize the system first from the sidebar.")
        return
    
    # Database statistics
    st.subheader("Database Statistics")
    
    try:
        db_stats = st.session_state.rag_pipeline.db_comparator.get_database_stats()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**ChromaDB**")
            if "chroma" in db_stats:
                st.json(db_stats["chroma"])
            else:
                st.warning("ChromaDB not available")
        
        with col2:
            st.markdown("**Pinecone**")
            if "pinecone" in db_stats:
                st.json(db_stats["pinecone"])
            else:
                st.warning("Pinecone not available")
    
    except Exception as e:
        st.error(f"Error getting database stats: {str(e)}")
    
    # Query comparison
    st.subheader("Query Comparison")
    
    with st.form("comparison_form"):
        comparison_query = st.text_input(
            "Enter a query to compare across databases:",
            value="What are fundamental human rights?"
        )
        
        k_value = st.slider("Number of results (k)", 1, 10, 5)
        
        compare_button = st.form_submit_button("🔍 Compare Databases")
    
    if compare_button and comparison_query:
        with st.spinner("Comparing databases..."):
            try:
                comparison_results = st.session_state.rag_pipeline.db_comparator.compare_retrieval(
                    comparison_query, k=k_value
                )
                
                # Display results
                for db_name, results in comparison_results.items():
                    st.subheader(f"📚 {db_name.title()} Results")
                    
                    if results.get('success', False):
                        st.success(f"Retrieved {len(results['documents'])} documents in {results['time_taken']:.3f}s")
                        
                        for i, doc in enumerate(results['documents']):
                            with st.expander(f"Document {i+1}"):
                                # Handle both Document objects and formatted dictionaries
                                if hasattr(doc, 'page_content'):
                                    content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                                elif isinstance(doc, dict) and 'content' in doc:
                                    content = doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content']
                                else:
                                    content = str(doc)[:500] + "..."
                                
                                st.text_area(
                                    "Content", 
                                    content,
                                    height=150,
                                    key=f"{db_name}_doc_{i}"
                                )
                                
                                # Handle metadata
                                if hasattr(doc, 'metadata') and getattr(doc, 'metadata', None):
                                    st.json(getattr(doc, 'metadata'))
                                elif isinstance(doc, dict) and 'metadata' in doc:
                                    st.json(doc['metadata'])
                    else:
                        st.error(f"Error: {results.get('error', 'Unknown error')}")
                
            except Exception as e:
                st.error(f"Comparison failed: {str(e)}")

def evaluation_dashboard_page():
    """Evaluation dashboard and analytics page."""
    
    st.header("🔬 Evaluation Dashboard")
    
    if not st.session_state.evaluation_suite:
        st.warning("Please initialize the system first from the sidebar.")
        return
    
    # Evaluation controls
    st.subheader("Run Evaluations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🧪 Run Benchmark", use_container_width=True):
            with st.spinner("Running benchmark evaluation..."):
                try:
                    benchmark_results = st.session_state.evaluation_suite.run_comprehensive_evaluation()
                    st.session_state.evaluation_results.append(benchmark_results)
                    st.success("Benchmark completed!")
                    st.json(benchmark_results)
                except Exception as e:
                    st.error(f"Benchmark failed: {str(e)}")
    
    with col2:
        if st.button("📊 Database Analysis", use_container_width=True):
            with st.spinner("Analyzing databases..."):
                try:
                    db_evaluation = st.session_state.evaluation_suite.evaluate_database_comparison()
                    st.session_state.evaluation_results.append(db_evaluation)
                    st.success("Database analysis completed!")
                    st.json(db_evaluation)
                except Exception as e:
                    st.error(f"Database analysis failed: {str(e)}")
    
    with col3:
        if st.button("📄 Generate Report", use_container_width=True):
            with st.spinner("Generating evaluation report..."):
                try:
                    report_path = st.session_state.evaluation_suite.generate_evaluation_report(
                        st.session_state.evaluation_results
                    )
                    st.success(f"Report generated: {report_path}")
                    
                    # Provide download link
                    if os.path.exists(report_path):
                        with open(report_path, 'r', encoding='utf-8') as f:
                            report_content = f.read()
                        
                        st.download_button(
                            label="📥 Download Report",
                            data=report_content,
                            file_name="evaluation_report.md",
                            mime="text/markdown"
                        )
                except Exception as e:
                    st.error(f"Report generation failed: {str(e)}")
    
    # Evaluation results visualization
    if st.session_state.evaluation_results:
        st.subheader("📈 Evaluation Results")
        
        # Display recent results
        for i, eval_result in enumerate(st.session_state.evaluation_results[-5:]):  # Show last 5
            with st.expander(f"Evaluation {i+1}"):
                st.json(eval_result)
    
    else:
        st.info("No evaluation results available. Run some evaluations to see analytics.")

def settings_page():
    """Settings and configuration page."""
    
    st.header("⚙️ Settings & Configuration")
    
    # API Configuration
    st.subheader("🔑 API Configuration")
    
    with st.expander("API Keys", expanded=False):
        st.info("Configure your API keys in the .env file")
        
        # Check API key status
        api_status = {
            "OpenAI": "✅ Configured" if Config.OPENAI_API_KEY else "❌ Missing",
            "Groq": "✅ Configured" if Config.GROQ_API_KEY else "❌ Missing",
            "Pinecone": "✅ Configured" if Config.PINECONE_API_KEY else "❌ Missing",
            "LangSmith": "✅ Configured" if Config.LANGCHAIN_API_KEY else "❌ Missing"
        }
        
        for service, status in api_status.items():
            st.text(f"{service}: {status}")
    
    # Model Configuration
    st.subheader("🤖 Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.text_input("Embedding Model", value=Config.EMBEDDING_MODEL, disabled=True)
        st.text_input("Default LLM", value=getattr(Config, "DEFAULT_LLM", "Not Configured"), disabled=True)
    
    with col2:
        st.text_input("Groq Model", value=Config.GROQ_MODEL, disabled=True)
        st.text_input("LangSmith Project", value=Config.LANGCHAIN_PROJECT, disabled=True)
    
    # System Information
    st.subheader("ℹ️ System Information")
    
    system_info = {
        "Python Environment": "Virtual Environment Active",
        "LangSmith Tracing": "Enabled" if Config.LANGCHAIN_TRACING_V2 == "true" else "Disabled",
        "ChromaDB Directory": Config.CHROMA_PERSIST_DIRECTORY,
        "Pinecone Index": Config.PINECONE_INDEX_NAME
    }
    
    for key, value in system_info.items():
        st.text(f"{key}: {value}")
    
    # Clear data
    st.subheader("🗑️ Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
    
    with col2:
        if st.button("Clear Evaluation Results", use_container_width=True):
            st.session_state.evaluation_results = []
            st.success("Evaluation results cleared!")

if __name__ == "__main__":
    main()
