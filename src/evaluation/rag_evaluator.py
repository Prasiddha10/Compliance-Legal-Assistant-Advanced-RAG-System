"""End-to-end RAG evaluation system."""
from typing import Dict, Any, List, Optional
import json
import time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .retrieval_eval import RetrievalEvaluator, RetrievalBenchmark
from .generation_eval import GenerationEvaluator
from .llm_judge import LLMJudge
from src.rag.pipeline import RAGPipeline
from src.database.comparator import DatabaseComparator
import logging

logger = logging.getLogger(__name__)

class RAGEvaluationSuite:
    """Comprehensive RAG evaluation suite."""
    
    def __init__(self, rag_pipeline: Optional[RAGPipeline] = None):
        self.rag_pipeline = rag_pipeline or RAGPipeline()
        self.retrieval_evaluator = RetrievalEvaluator()
        self.generation_evaluator = GenerationEvaluator()
        self.llm_judge = LLMJudge()
        self.benchmark = RetrievalBenchmark()
        
        # Results storage
        self.evaluation_results = []
        
        logger.info("RAG Evaluation Suite initialized")
    
    def evaluate_single_query(self, query: str, reference_answer: Optional[str] = None, 
                             ground_truth_docs: Optional[List] = None) -> Dict[str, Any]:
        """Evaluate a single query through the RAG pipeline."""
        
        # Record start time
        start_time = time.time()
        
        # Get RAG response
        rag_result = self.rag_pipeline.query(query)
        
        # Record processing time
        processing_time = time.time() - start_time
        
        # Initialize evaluation result
        evaluation = {
            "query": query,
            "timestamp": time.time(),
            "processing_time": processing_time,
            "rag_result": rag_result
        }
        
        # Retrieval evaluation
        if rag_result.get("retrieved_docs"):
            from langchain_core.documents.base import Document
            retrieved_docs = [
                Document(page_content=doc['content'], metadata=doc['metadata'])
                for doc in rag_result["retrieved_docs"]
            ]
            
            retrieval_metrics = self.retrieval_evaluator.evaluate_retrieval_quality(
                query, retrieved_docs, ground_truth_docs or []
            )
            evaluation["retrieval_metrics"] = retrieval_metrics
        
        # Generation evaluation
        if rag_result.get("response"):
            context = rag_result.get("context", "")
            
            generation_metrics = self.generation_evaluator.evaluate_generation_quality(
                rag_result["response"], query, context, reference_answer
            )
            evaluation["generation_metrics"] = generation_metrics
            
            # LLM Judge evaluation
            judge_evaluation = self.llm_judge.comprehensive_evaluation(
                query, context, rag_result["response"]
            )
            evaluation["judge_evaluation"] = judge_evaluation
        
        # Overall performance score
        evaluation["overall_performance"] = self._calculate_overall_performance(evaluation)
        
        # Store result
        self.evaluation_results.append(evaluation)
        
        logger.info(f"Evaluated query: {query[:50]}... Score: {evaluation['overall_performance']:.2f}")
        
        return evaluation
    
    def evaluate_response(self, query: str, response: str, context: str = "", 
                         reference_answer: Optional[str] = None, 
                         ground_truth_docs: Optional[List] = None) -> Dict[str, Any]:
        """Evaluate a pre-generated response without re-querying the RAG pipeline."""
        
        # Initialize evaluation result
        evaluation = {
            "query": query,
            "timestamp": time.time(),
            "processing_time": 0.0,  # Not applicable for pre-generated responses
            "rag_result": {
                "response": response,
                "context": context,
                "retrieved_docs": []
            }
        }
        
        # Generation evaluation
        if response:
            generation_metrics = self.generation_evaluator.evaluate_generation_quality(
                response, query, context, reference_answer
            )
            evaluation["generation_metrics"] = generation_metrics
            
            # LLM Judge evaluation
            judge_evaluation = self.llm_judge.comprehensive_evaluation(
                query, context, response
            )
            evaluation["judge_evaluation"] = judge_evaluation
        
        # Overall performance score
        evaluation["overall_performance"] = self._calculate_overall_performance(evaluation)
        
        logger.info(f"Evaluated response for query: {query[:50]}... Score: {evaluation['overall_performance']:.2f}")
        
        return evaluation
    
    def run_benchmark_evaluation(self, save_results: bool = True) -> Dict[str, Any]:
        """Run benchmark evaluation on predefined test queries."""
        logger.info("Starting benchmark evaluation...")
        
        benchmark_results = {
            "start_time": time.time(),
            "test_queries": [],
            "summary_metrics": {}
        }
        
        # Run benchmark
        for test_query in self.benchmark.test_queries:
            query = test_query["query"]
            reference_answer = test_query.get("reference_answer")
            
            # Evaluate query with reference answer for ROUGE/BLEU scoring
            evaluation = self.evaluate_single_query(query, reference_answer)
            
            # Add test-specific metadata
            evaluation.update({
                "expected_topics": test_query["expected_topics"],
                "difficulty": test_query["difficulty"]
            })
            
            benchmark_results["test_queries"].append(evaluation)
        
        # Calculate summary metrics
        benchmark_results["summary_metrics"] = self._calculate_benchmark_summary(
            benchmark_results["test_queries"]
        )
        
        benchmark_results["end_time"] = time.time()
        benchmark_results["total_time"] = benchmark_results["end_time"] - benchmark_results["start_time"]
        
        if save_results:
            self._save_benchmark_results(benchmark_results)
        
        logger.info("Benchmark evaluation completed")
        return benchmark_results
    
    def run_comprehensive_evaluation(self, save_results: bool = True) -> Dict[str, Any]:
        """Run a comprehensive evaluation including benchmark and database comparison."""
        logger.info("Starting comprehensive evaluation...")
        
        start_time = time.time()
        
        # Run benchmark evaluation
        benchmark_results = self.run_benchmark_evaluation(save_results=False)
        
        # Run database comparison evaluation
        db_comparison_results = self.evaluate_database_comparison()
        
        # Combine results
        comprehensive_results = {
            "evaluation_type": "comprehensive",
            "timestamp": time.time(),
            "total_evaluation_time": time.time() - start_time,
            "benchmark_results": benchmark_results,
            "database_comparison": db_comparison_results,
            "overall_summary": {
                "benchmark_performance": benchmark_results.get("summary", {}),
                "database_performance": db_comparison_results.get("summary", {}),
                "total_queries_evaluated": len(benchmark_results.get("query_evaluations", [])),
                "databases_compared": len(db_comparison_results.get("database_comparisons", []))
            }
        }
        
        # Calculate comprehensive score
        benchmark_score = benchmark_results.get("summary", {}).get("overall_performance", 0)
        db_score = db_comparison_results.get("summary", {}).get("average_performance", 0)
        comprehensive_results["comprehensive_score"] = (benchmark_score + db_score) / 2
        
        # Save results if requested
        if save_results:
            self._save_comprehensive_results(comprehensive_results)
        
        # Add to evaluation results
        self.evaluation_results.append(comprehensive_results)
        
        logger.info(f"Comprehensive evaluation completed in {comprehensive_results['total_evaluation_time']:.2f}s")
        return comprehensive_results
    
    def _save_comprehensive_results(self, results: Dict[str, Any]):
        """Save comprehensive evaluation results."""
        try:
            results_dir = Path("evaluation_results")
            results_dir.mkdir(exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"comprehensive_eval_{timestamp}.json"
            
            with open(results_dir / filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Comprehensive results saved to {results_dir / filename}")
        except Exception as e:
            logger.error(f"Error saving comprehensive results: {e}")
    
    def evaluate_database_comparison(self) -> Dict[str, Any]:
        """Evaluate and compare different vector databases."""
        logger.info("Evaluating database comparison...")
        
        db_comparator = DatabaseComparator()
        
        # Get database statistics
        db_stats = db_comparator.get_database_stats()
        
        # Test queries for comparison
        test_queries = [
            "What are fundamental human rights?",
            "How does international law protect refugees?",
            "What are the obligations of states regarding freedom of expression?"
        ]
        
        comparison_results = {
            "database_stats": db_stats,
            "query_comparisons": []
        }
        
        for query in test_queries:
            try:
                comparison = db_comparator.compare_search_results(query, k=5)
                comparison_results["query_comparisons"].append(comparison)
            except Exception as e:
                logger.error(f"Error comparing databases for query '{query}': {e}")
        
        # Calculate aggregate comparison metrics
        comparison_results["aggregate_metrics"] = self._calculate_db_comparison_metrics(
            comparison_results["query_comparisons"]
        )
        
        return comparison_results
    
    def generate_evaluation_report(self, evaluation_results: Optional[List[Dict[str, Any]]] = None, output_dir: str = "evaluation_reports") -> str:
        """Generate comprehensive evaluation report."""
        logger.info("Generating evaluation report...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Use provided results or run evaluation if none exist
        if evaluation_results:
            self.evaluation_results.extend(evaluation_results)
        elif not self.evaluation_results:
            self.run_benchmark_evaluation()
        
        # Generate visualizations
        self._create_evaluation_visualizations(output_path)
        
        # Generate HTML report
        report_path = self._generate_html_report(output_path)
        
        logger.info(f"Evaluation report generated: {report_path}")
        return str(report_path)
    
    def _calculate_overall_performance(self, evaluation: Dict[str, Any]) -> float:
        """Calculate overall performance score."""
        scores = []
        weights = []
        
        # Retrieval performance (weight: 0.3)
        if "retrieval_metrics" in evaluation:
            retrieval_score = evaluation["retrieval_metrics"].get("avg_score", 0.5)
            scores.append(retrieval_score)
            weights.append(0.3)
        
        # Generation performance (weight: 0.4)
        if "generation_metrics" in evaluation:
            gen_score = evaluation["generation_metrics"].get("overall_quality", 0.5)
            scores.append(gen_score)
            weights.append(0.4)
        
        # Judge evaluation (weight: 0.3)
        if "judge_evaluation" in evaluation:
            judge_score = evaluation["judge_evaluation"].get("comprehensive_score", 5.0) / 10.0
            scores.append(judge_score)
            weights.append(0.3)
        
        if not scores:
            return 0.0
        
        # Weighted average
        weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
        total_weight = sum(weights)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _calculate_benchmark_summary(self, query_evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate benchmark summary metrics."""
        if not query_evaluations:
            return {}
        
        # Overall performance
        overall_scores = [eval.get("overall_performance", 0) for eval in query_evaluations]
        
        # Processing times
        processing_times = [eval.get("processing_time", 0) for eval in query_evaluations]
        
        # Success rate
        successful_queries = [eval for eval in query_evaluations if not eval.get("rag_result", {}).get("error")]
        success_rate = len(successful_queries) / len(query_evaluations)
        
        # Performance by difficulty
        difficulty_performance = {}
        for difficulty in ["easy", "medium", "hard"]:
            difficulty_evals = [eval for eval in query_evaluations if eval.get("difficulty") == difficulty]
            if difficulty_evals:
                avg_score = sum(eval.get("overall_performance", 0) for eval in difficulty_evals) / len(difficulty_evals)
                difficulty_performance[difficulty] = avg_score
        
        return {
            "total_queries": len(query_evaluations),
            "success_rate": success_rate,
            "avg_overall_performance": sum(overall_scores) / len(overall_scores),
            "avg_processing_time": sum(processing_times) / len(processing_times),
            "performance_by_difficulty": difficulty_performance,
            "min_performance": min(overall_scores),
            "max_performance": max(overall_scores)
        }
    
    def _calculate_db_comparison_metrics(self, comparisons: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate database comparison metrics."""
        if not comparisons:
            return {}
        
        # Average overlap ratios
        overlap_ratios = [comp.get("comparison_metrics", {}).get("overlap_ratio", 0) for comp in comparisons]
        jaccard_similarities = [comp.get("comparison_metrics", {}).get("jaccard_similarity", 0) for comp in comparisons]
        
        # Average scores by database
        chroma_scores = []
        pinecone_scores = []
        
        for comp in comparisons:
            chroma_results = comp.get("chroma_results", [])
            pinecone_results = comp.get("pinecone_results", [])
            
            if chroma_results:
                avg_chroma = sum(r.get("score", 0) for r in chroma_results) / len(chroma_results)
                chroma_scores.append(avg_chroma)
            
            if pinecone_results:
                avg_pinecone = sum(r.get("score", 0) for r in pinecone_results) / len(pinecone_results)
                pinecone_scores.append(avg_pinecone)
        
        return {
            "avg_overlap_ratio": sum(overlap_ratios) / len(overlap_ratios) if overlap_ratios else 0,
            "avg_jaccard_similarity": sum(jaccard_similarities) / len(jaccard_similarities) if jaccard_similarities else 0,
            "avg_chroma_score": sum(chroma_scores) / len(chroma_scores) if chroma_scores else 0,
            "avg_pinecone_score": sum(pinecone_scores) / len(pinecone_scores) if pinecone_scores else 0,
            "chroma_queries": len(chroma_scores),
            "pinecone_queries": len(pinecone_scores)
        }
    
    def _save_benchmark_results(self, results: Dict[str, Any]):
        """Save benchmark results to JSON file."""
        timestamp = int(time.time())
        filename = f"benchmark_results_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Benchmark results saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving benchmark results: {e}")
    
    def _create_evaluation_visualizations(self, output_path: Path):
        """Create evaluation visualizations."""
        if not self.evaluation_results:
            return
        
        # Performance over time
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Overall Performance', 'Processing Time', 'Retrieval Metrics', 'Generation Metrics')
        )
        
        # Overall performance
        performances = [eval.get("overall_performance", 0) for eval in self.evaluation_results]
        fig.add_trace(
            go.Scatter(y=performances, mode='lines+markers', name='Overall Performance'),
            row=1, col=1
        )
        
        # Processing time
        times = [eval.get("processing_time", 0) for eval in self.evaluation_results]
        fig.add_trace(
            go.Scatter(y=times, mode='lines+markers', name='Processing Time'),
            row=1, col=2
        )
        
        # Save plot
        fig.write_html(str(output_path / "performance_dashboard.html"))
        
        # Create performance distribution
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.hist(performances, bins=20, alpha=0.7)
        plt.title('Performance Distribution')
        plt.xlabel('Overall Performance Score')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 2, 2)
        plt.hist(times, bins=20, alpha=0.7)
        plt.title('Processing Time Distribution')
        plt.xlabel('Processing Time (seconds)')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(output_path / "performance_distributions.png")
        plt.close()
    
    def _generate_html_report(self, output_path: Path) -> Path:
        """Generate HTML evaluation report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RAG Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .metric {{ background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                .score {{ font-weight: bold; color: #2E8B57; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>RAG System Evaluation Report</h1>
            <p>Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Summary</h2>
            <div class="metric">
                <strong>Total Queries Evaluated:</strong> {len(self.evaluation_results)}
            </div>
            
            {self._generate_summary_html()}
            
            <h2>Detailed Results</h2>
            {self._generate_detailed_results_html()}
            
            <h2>Visualizations</h2>
            <p><a href="performance_dashboard.html">Interactive Performance Dashboard</a></p>
            <img src="performance_distributions.png" alt="Performance Distributions" style="max-width: 100%;">
            
        </body>
        </html>
        """
        
        report_path = output_path / "evaluation_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return report_path
    
    def _generate_summary_html(self) -> str:
        """Generate summary HTML section."""
        if not self.evaluation_results:
            return "<p>No evaluation results available.</p>"
        
        avg_performance = sum(eval.get("overall_performance", 0) for eval in self.evaluation_results) / len(self.evaluation_results)
        avg_time = sum(eval.get("processing_time", 0) for eval in self.evaluation_results) / len(self.evaluation_results)
        
        return f"""
        <div class="metric">
            <strong>Average Overall Performance:</strong> <span class="score">{avg_performance:.3f}</span>
        </div>
        <div class="metric">
            <strong>Average Processing Time:</strong> {avg_time:.2f} seconds
        </div>
        """
    
    def _generate_detailed_results_html(self) -> str:
        """Generate detailed results HTML table."""
        if not self.evaluation_results:
            return "<p>No detailed results available.</p>"
        
        html = """
        <table>
            <tr>
                <th>Query</th>
                <th>Overall Performance</th>
                <th>Processing Time</th>
                <th>Database Used</th>
                <th>Judge Score</th>
            </tr>
        """
        
        for eval in self.evaluation_results:
            query = eval.get("query", "")[:50] + "..." if len(eval.get("query", "")) > 50 else eval.get("query", "")
            performance = eval.get("overall_performance", 0)
            time_taken = eval.get("processing_time", 0)
            db_used = eval.get("rag_result", {}).get("database_used", "unknown")
            judge_score = eval.get("judge_evaluation", {}).get("comprehensive_score", 0)
            
            html += f"""
            <tr>
                <td>{query}</td>
                <td>{performance:.3f}</td>
                <td>{time_taken:.2f}s</td>
                <td>{db_used}</td>
                <td>{judge_score:.2f}</td>
            </tr>
            """
        
        html += "</table>"
        return html
