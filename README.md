# Human Rights Legal Assistant - RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system specifically designed for human rights law queries, built with LangGraph, monitored by LangSmith, and featuring extensive evaluation capabilities.

## 🌟 Features

### Core Functionality
- **Advanced RAG Pipeline**: Built with LangGraph for sophisticated workflow orchestration
- **Multi-Database Support**: Compare ChromaDB vs Pinecone performance side-by-side
- **Legal Document Processing**: Specialized PDF processing for human rights documents
- **Intelligent Query Processing**: Natural language queries for legal information

### Monitoring & Evaluation
- **LangSmith Integration**: Real-time monitoring and tracing of all operations
- **Comprehensive Evaluation**: Retrieval, generation, and end-to-end metrics
- **LLM as Judge**: AI-powered evaluation of response quality
- **Performance Analytics**: Detailed dashboards and visualizations

### AI Models
- **Multiple LLM Providers**: OpenAI GPT models + Groq (Mixtral, Llama2)
- **State-of-the-art Embeddings**: Sentence Transformers for semantic search
- **Fallback Systems**: Robust handling of API failures

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │    CLI Interface │    │  Jupyter Notebooks │
└─────────┬───────┘    └─────────┬────────┘    └─────────┬─────────┘
          │                      │                       │
          └──────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────┴──────────────┐
                    │      RAG Pipeline          │
                    │      (LangGraph)           │
                    └─────────────┬──────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
    ┌─────────▼─────────┐ ┌──────▼──────┐ ┌─────────▼─────────┐
    │   ChromaDB        │ │  Pinecone   │ │  LLM Manager      │
    │   (Local)         │ │  (Cloud)    │ │  (OpenAI/Groq)    │
    └───────────────────┘ └─────────────┘ └───────────────────┘
              │                  │                  │
              └──────────────────┼──────────────────┘
                                 │
                    ┌─────────────▼──────────────┐
                    │     Evaluation Suite       │
                    │   (Metrics + LLM Judge)    │
                    └────────────────────────────┘
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
cd "c:\Users\Prasiddha\Downloads\Lang"

# Virtual environment is already configured
# Install dependencies (already done)
```

### 2. Configure API Keys

Edit the `.env` file with your API keys:

```env
# Required API Keys
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# Optional but recommended
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENV=your_pinecone_environment_here
LANGSMITH_API_KEY=your_langsmith_api_key_here

# LangSmith Configuration (for monitoring)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_PROJECT=rag-human-rights-evaluation
```

### 3. Run the Application

#### Streamlit Web Interface (Recommended)
```bash
streamlit run app.py
```

#### Command Line Interface
```bash
# Query the system
python cli.py query "What are the fundamental principles of human rights?"

# Upload documents
python cli.py upload path/to/human_rights_document.pdf

# Run evaluation benchmark
python cli.py evaluate

# Test system components
python cli.py test
```

## 📚 Usage Examples

### 1. Upload Human Rights Documents

```python
from src.rag.pipeline import RAGPipeline

# Initialize the system
rag = RAGPipeline()

# Upload and process a document
result = rag.add_documents("path/to/universal_declaration.pdf")
print(f"Processed {result['num_documents']} document chunks")
```

### 2. Query the System

```python
# Ask a legal question
response = rag.query("What does Article 19 of the UDHR say about freedom of expression?")

print(f"Answer: {response['response']}")
print(f"Sources: {len(response['retrieved_docs'])} documents")
print(f"Database used: {response['database_used']}")
```

### 3. Run Comprehensive Evaluation

```python
from src.evaluation.rag_evaluator import RAGEvaluationSuite

# Initialize evaluation suite
evaluator = RAGEvaluationSuite(rag)

# Run benchmark evaluation
results = evaluator.run_benchmark_evaluation()

# Generate detailed report
report_path = evaluator.generate_evaluation_report()
print(f"Report generated: {report_path}")
```

### 4. Compare Database Performance

```python
from src.database.comparator import DatabaseComparator

# Compare ChromaDB vs Pinecone
comparator = DatabaseComparator()
comparison = comparator.compare_search_results(
    "refugee protection under international law", 
    k=5
)

print(f"Overlap ratio: {comparison['comparison_metrics']['overlap_ratio']}")
print(f"ChromaDB results: {len(comparison['chroma_results'])}")
print(f"Pinecone results: {len(comparison['pinecone_results'])}")
```

## 📊 Evaluation Metrics

### Retrieval Metrics
- **Precision@K**: Proportion of relevant documents in top-k results
- **Recall@K**: Proportion of relevant documents retrieved
- **F1@K**: Harmonic mean of precision and recall
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **MRR**: Mean Reciprocal Rank
- **Hit Rate@K**: Proportion of queries with at least one relevant result

### Generation Metrics
- **ROUGE Scores**: Text overlap with reference answers
- **BLEU Score**: N-gram overlap evaluation
- **Factual Accuracy**: Fact verification against source documents
- **Coherence**: Text flow and logical consistency
- **Relevance**: Alignment with query intent
- **Fluency**: Language quality and readability

### LLM Judge Metrics
- **Overall Quality**: Comprehensive response assessment
- **Legal Accuracy**: Correctness of legal information
- **Context Utilization**: Effective use of retrieved documents
- **Hallucination Detection**: Identification of unsupported claims

## 🔧 Configuration

### Model Configuration

The system supports multiple embedding and LLM models:

```python
# Embedding Models
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Fast, efficient
# Alternative: "sentence-transformers/all-mpnet-base-v2"     # Higher quality

# LLM Models
DEFAULT_LLM_MODEL = "gpt-3.5-turbo"    # OpenAI
GROQ_MODEL = "mixtral-8x7b-32768"      # Groq (fast inference)
# Alternative: "llama2-70b-4096"        # Groq (high quality)
```

### Database Configuration

```python
# ChromaDB (Local)
CHROMA_PERSIST_DIRECTORY = "./data/chroma_db"

# Pinecone (Cloud)
PINECONE_INDEX_NAME = "human-rights-index"
PINECONE_ENV = "us-west1-gcp"  # or your preferred environment
```

## 📁 Project Structure

```
Lang/
├── src/
│   ├── rag/                    # RAG pipeline and LLM management
│   │   ├── pipeline.py         # LangGraph-based RAG workflow
│   │   └── llm_manager.py      # Multi-provider LLM management
│   ├── database/               # Vector database management
│   │   ├── chroma_db.py        # ChromaDB implementation
│   │   ├── pinecone_db.py      # Pinecone implementation
│   │   └── comparator.py       # Database comparison tools
│   ├── evaluation/             # Comprehensive evaluation suite
│   │   ├── retrieval_eval.py   # Retrieval metrics
│   │   ├── generation_eval.py  # Generation metrics
│   │   ├── llm_judge.py        # LLM-as-judge evaluation
│   │   └── rag_evaluator.py    # End-to-end evaluation
│   ├── utils/                  # Utility functions
│   │   ├── pdf_processor.py    # PDF processing and chunking
│   │   └── embeddings.py       # Embedding utilities
│   └── config.py               # Configuration management
├── app.py                      # Streamlit web interface
├── cli.py                      # Command-line interface
├── requirements.txt            # Python dependencies
└── .env                        # Environment configuration
```

## 🎯 Use Cases

### Legal Research
- Query specific human rights laws and conventions
- Find relevant case law and precedents
- Compare different legal frameworks

### Educational Tool
- Learn about human rights principles
- Understand legal terminology and concepts
- Access structured legal information

### Compliance Analysis
- Check organizational policies against human rights standards
- Identify potential compliance gaps
- Get guidance on best practices

## 🔍 Advanced Features

### LangGraph Workflow
The RAG pipeline uses LangGraph for sophisticated workflow orchestration:

1. **Document Retrieval**: Multi-database search with scoring
2. **Document Reranking**: Relevance-based result optimization
3. **Response Generation**: Context-aware answer generation
4. **Quality Evaluation**: Automatic response assessment

### LangSmith Monitoring
All operations are traced and monitored through LangSmith:

- **Request Tracing**: Full pipeline visibility
- **Performance Metrics**: Latency and throughput monitoring
- **Error Tracking**: Automated issue detection
- **Usage Analytics**: Cost and usage optimization

### Database Comparison
Real-time comparison of vector database performance:

- **Accuracy Comparison**: Retrieval quality metrics
- **Speed Benchmarks**: Query response times
- **Cost Analysis**: Resource utilization
- **Reliability Metrics**: Uptime and error rates

## 🧪 Testing

### Run System Tests
```bash
# Test all components
python cli.py test --component all

# Test specific components
python cli.py test --component llm
python cli.py test --component database
```

### Benchmark Evaluation
```bash
# Run full benchmark suite
python cli.py evaluate

# Custom evaluation with specific queries
python -c "
from src.evaluation.rag_evaluator import RAGEvaluationSuite
from src.rag.pipeline import RAGPipeline

rag = RAGPipeline()
evaluator = RAGEvaluationSuite(rag)
result = evaluator.evaluate_single_query('Your custom legal question here')
print(f'Performance: {result[\"overall_performance\"]:.3f}')
"
```

## 📈 Performance Optimization

### Embedding Optimization
- Use GPU acceleration for faster embedding computation
- Batch process documents for improved throughput
- Cache embeddings to avoid recomputation

### Database Optimization
- Optimize vector index parameters for your use case
- Use appropriate similarity metrics (cosine, euclidean, dot product)
- Consider database sharding for large document collections

### LLM Optimization
- Use streaming responses for better user experience
- Implement response caching for common queries
- Balance model quality vs. speed based on requirements

## 🛠️ Troubleshooting

### Common Issues

#### API Key Errors
```bash
# Check API key configuration
python -c "from src.config import Config; Config.validate_config()"
```

#### Database Connection Issues
```bash
# Test database connectivity
python cli.py test --component database
```

#### Model Loading Issues
```bash
# Test LLM models
python cli.py test --component llm
```

### Performance Issues
- **Slow queries**: Check vector database index size and parameters
- **High memory usage**: Reduce batch sizes or use smaller models
- **API rate limits**: Implement exponential backoff and request queuing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Update documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LangChain**: For the foundational RAG framework
- **LangGraph**: For workflow orchestration capabilities
- **Sentence Transformers**: For high-quality embeddings
- **Human Rights Organizations**: For providing valuable legal documents and expertise

## 📞 Support

For questions, issues, or contributions:

1. **Documentation**: Check this README and inline code documentation
2. **Issues**: Create a GitHub issue with detailed description
3. **Discussions**: Use GitHub Discussions for general questions
4. **Email**: Contact the development team for urgent matters

---

**Built with ❤️ for advancing human rights knowledge and accessibility**
