"""Command-line interface for the RAG system."""
import argparse
import sys
import json
from pathlib import Path

from src.rag.pipeline import RAGPipeline
from src.evaluation.rag_evaluator import RAGEvaluationSuite
from src.utils.pdf_processor import PDFProcessor
from src.config import Config

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Human Rights RAG System CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the RAG system')
    query_parser.add_argument('question', help='Question to ask')
    query_parser.add_argument('--evaluate', action='store_true', help='Run evaluation')
    
    # Upload command
    upload_parser = subparsers.add_parser('upload', help='Upload and process documents')
    upload_parser.add_argument('files', nargs='+', help='PDF files to upload')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Run evaluation benchmark')
    eval_parser.add_argument('--output', default='evaluation_results.json', help='Output file')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test system components')
    test_parser.add_argument('--component', choices=['llm', 'database', 'all'], default='all')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        # Validate configuration
        Config.validate_config()
        
        # Initialize system
        print("Initializing RAG system...")
        rag_pipeline = RAGPipeline()
        
        if args.command == 'query':
            handle_query(rag_pipeline, args)
        elif args.command == 'upload':
            handle_upload(rag_pipeline, args)
        elif args.command == 'evaluate':
            handle_evaluate(rag_pipeline, args)
        elif args.command == 'test':
            handle_test(rag_pipeline, args)
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

def handle_query(rag_pipeline, args):
    """Handle query command."""
    print(f"Processing query: {args.question}")
    
    result = rag_pipeline.query(args.question)
    
    print("\n" + "="*50)
    print("QUERY RESULT")
    print("="*50)
    print(f"Question: {result['query']}")
    print(f"Answer: {result['response']}")
    print(f"Database Used: {result['database_used']}")
    print(f"Documents Retrieved: {len(result['retrieved_docs'])}")
    
    if result.get('error'):
        print(f"Error: {result['error']}")
    
    if args.evaluate:
        print("\nRunning evaluation...")
        evaluation_suite = RAGEvaluationSuite(rag_pipeline)
        evaluation = evaluation_suite.evaluate_single_query(args.question)
        
        print(f"Overall Performance: {evaluation['overall_performance']:.3f}")
        print(f"Processing Time: {evaluation['processing_time']:.2f}s")

def handle_upload(rag_pipeline, args):
    """Handle upload command."""
    print(f"Processing {len(args.files)} files...")
    
    for file_path in args.files:
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"Warning: File not found: {file_path}")
            continue
        
        if file_path.suffix.lower() != '.pdf':
            print(f"Warning: Not a PDF file: {file_path}")
            continue
        
        print(f"Processing: {file_path.name}")
        
        try:
            result = rag_pipeline.add_documents(str(file_path))
            print(f"✅ Processed {result['num_documents']} chunks")
            
            if result.get('error'):
                print(f"❌ Error: {result['error']}")
                
        except Exception as e:
            print(f"❌ Error processing {file_path.name}: {str(e)}")

def handle_evaluate(rag_pipeline, args):
    """Handle evaluate command."""
    print("Running benchmark evaluation...")
    
    evaluation_suite = RAGEvaluationSuite(rag_pipeline)
    results = evaluation_suite.run_benchmark_evaluation()
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    summary = results['summary_metrics']
    print(f"Total Queries: {summary['total_queries']}")
    print(f"Success Rate: {summary['success_rate']:.2%}")
    print(f"Average Performance: {summary['avg_overall_performance']:.3f}")
    print(f"Average Processing Time: {summary['avg_processing_time']:.2f}s")
    
    # Performance by difficulty
    if 'performance_by_difficulty' in summary:
        print("\nPerformance by Difficulty:")
        for difficulty, score in summary['performance_by_difficulty'].items():
            print(f"  {difficulty.title()}: {score:.3f}")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {args.output}")

def handle_test(rag_pipeline, args):
    """Handle test command."""
    print(f"Testing {args.component} components...")
    
    if args.component in ['llm', 'all']:
        print("\n📋 Testing LLM Models:")
        llm_manager = rag_pipeline.llm_manager
        
        for model_name in llm_manager.list_available_models():
            result = llm_manager.test_model(model_name)
            
            if result['success']:
                print(f"✅ {model_name}: {result['response_time']:.2f}s")
            else:
                print(f"❌ {model_name}: {result['error']}")
    
    if args.component in ['database', 'all']:
        print("\n📊 Testing Databases:")
        db_stats = rag_pipeline.db_comparator.get_database_stats()
        
        for db_name, stats in db_stats.items():
            if 'error' in stats:
                print(f"❌ {db_name}: {stats['error']}")
            else:
                doc_count = stats.get('document_count', stats.get('total_vector_count', 0))
                print(f"✅ {db_name}: {doc_count} documents/vectors")
    
    print("\n🧪 Running Test Query:")
    test_query = "What are fundamental human rights?"
    result = rag_pipeline.query(test_query)
    
    if result.get('error'):
        print(f"❌ Test query failed: {result['error']}")
    else:
        print(f"✅ Test query successful")
        print(f"   Response length: {len(result['response'])} characters")
        print(f"   Documents retrieved: {len(result['retrieved_docs'])}")
        print(f"   Database used: {result['database_used']}")

if __name__ == "__main__":
    main()
