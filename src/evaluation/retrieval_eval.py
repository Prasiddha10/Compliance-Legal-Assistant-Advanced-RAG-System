"""Retrieval evaluation metrics and tools."""
from typing import List, Dict, Any, Tuple
import numpy as np
from langchain.schema import Document
from src.utils.embeddings import EmbeddingUtils
import logging

logger = logging.getLogger(__name__)

class RetrievalEvaluator:
    """Evaluate retrieval performance metrics."""
    
    def __init__(self):
        self.embedding_utils = EmbeddingUtils()
    
    def calculate_precision_at_k(self, retrieved_docs: List[Document], 
                                relevant_docs: List[Document], k: int = 5) -> float:
        """Calculate Precision@K for retrieval."""
        if not retrieved_docs or k <= 0:
            return 0.0
        
        top_k_docs = retrieved_docs[:k]
        relevant_ids = {doc.metadata.get('doc_id', id(doc)) for doc in relevant_docs}
        retrieved_ids = {doc.metadata.get('doc_id', id(doc)) for doc in top_k_docs}
        
        relevant_retrieved = len(relevant_ids & retrieved_ids)
        return relevant_retrieved / min(k, len(top_k_docs))
    
    def calculate_recall_at_k(self, retrieved_docs: List[Document], 
                             relevant_docs: List[Document], k: int = 5) -> float:
        """Calculate Recall@K for retrieval."""
        if not relevant_docs:
            return 0.0
        
        top_k_docs = retrieved_docs[:k]
        relevant_ids = {doc.metadata.get('doc_id', id(doc)) for doc in relevant_docs}
        retrieved_ids = {doc.metadata.get('doc_id', id(doc)) for doc in top_k_docs}
        
        relevant_retrieved = len(relevant_ids & retrieved_ids)
        return relevant_retrieved / len(relevant_docs)
    
    def calculate_f1_at_k(self, retrieved_docs: List[Document], 
                         relevant_docs: List[Document], k: int = 5) -> float:
        """Calculate F1@K for retrieval."""
        precision = self.calculate_precision_at_k(retrieved_docs, relevant_docs, k)
        recall = self.calculate_recall_at_k(retrieved_docs, relevant_docs, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def calculate_mrr(self, query_results: List[Tuple[List[Document], List[Document]]]) -> float:
        """Calculate Mean Reciprocal Rank."""
        reciprocal_ranks = []
        
        for retrieved_docs, relevant_docs in query_results:
            relevant_ids = {doc.metadata.get('doc_id', id(doc)) for doc in relevant_docs}
            
            rank = 0
            for i, doc in enumerate(retrieved_docs):
                doc_id = doc.metadata.get('doc_id', id(doc))
                if doc_id in relevant_ids:
                    rank = 1 / (i + 1)
                    break
            
            reciprocal_ranks.append(rank)
        
        return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    
    def calculate_ndcg_at_k(self, retrieved_docs: List[Document], 
                           relevance_scores: List[float], k: int = 5) -> float:
        """Calculate Normalized Discounted Cumulative Gain@K."""
        if not retrieved_docs or not relevance_scores:
            return 0.0
        
        # DCG calculation
        dcg = 0.0
        for i in range(min(k, len(relevance_scores))):
            dcg += relevance_scores[i] / np.log2(i + 2)
        
        # IDCG calculation (ideal DCG)
        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg = 0.0
        for i in range(min(k, len(ideal_scores))):
            idcg += ideal_scores[i] / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def calculate_hit_rate_at_k(self, query_results: List[Tuple[List[Document], List[Document]]], 
                               k: int = 5) -> float:
        """Calculate Hit Rate@K."""
        hits = 0
        total_queries = len(query_results)
        
        for retrieved_docs, relevant_docs in query_results:
            relevant_ids = {doc.metadata.get('doc_id', id(doc)) for doc in relevant_docs}
            retrieved_ids = {doc.metadata.get('doc_id', id(doc)) for doc in retrieved_docs[:k]}
            
            if relevant_ids & retrieved_ids:
                hits += 1
        
        return hits / total_queries if total_queries > 0 else 0.0
    
    from typing import Optional

    def evaluate_retrieval_quality(self, query: str, retrieved_docs: List[Document], 
                                  relevant_docs: Optional[List[Document]] = None, 
                                  retrieval_scores: 'Optional[List[float]]' = None) -> Dict[str, Any]:
        """Comprehensive retrieval evaluation."""
        metrics = {
            "query": query,
            "num_retrieved": len(retrieved_docs),
            "num_relevant": len(relevant_docs) if relevant_docs else 0
        }
        
        # If we have ground truth relevant documents
        if relevant_docs:
            for k in [1, 3, 5, 10]:
                if k <= len(retrieved_docs):
                    metrics[f"precision@{k}"] = self.calculate_precision_at_k(
                        retrieved_docs, relevant_docs, k
                    )
                    metrics[f"recall@{k}"] = self.calculate_recall_at_k(
                        retrieved_docs, relevant_docs, k
                    )
                    metrics[f"f1@{k}"] = self.calculate_f1_at_k(
                        retrieved_docs, relevant_docs, k
                    )
        
        # If we have retrieval scores
        if retrieval_scores:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Retrieval scores for query '{query[:50]}...': {retrieval_scores}")
            metrics["avg_score"] = np.mean(retrieval_scores)
            metrics["max_score"] = np.max(retrieval_scores)
            metrics["min_score"] = np.min(retrieval_scores)
            metrics["score_std"] = np.std(retrieval_scores)
            
            # Use scores as relevance for NDCG
            for k in [1, 3, 5, 10]:
                if k <= len(retrieval_scores):
                    metrics[f"ndcg@{k}"] = self.calculate_ndcg_at_k(
                        retrieved_docs, retrieval_scores[:k], k
                    )
        
        # Diversity metrics
        if len(retrieved_docs) > 1:
            metrics["diversity_score"] = self._calculate_diversity(retrieved_docs)
        
        return metrics
    
    def _calculate_diversity(self, documents: List[Document]) -> float:
        """Calculate diversity of retrieved documents."""
        if len(documents) < 2:
            return 0.0
        
        # Calculate pairwise similarity and return 1 - average similarity
        similarities = []
        
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                # Simple diversity based on content overlap
                doc1_words = set(documents[i].page_content.lower().split())
                doc2_words = set(documents[j].page_content.lower().split())
                
                if len(doc1_words) == 0 or len(doc2_words) == 0:
                    similarity = 0.0
                else:
                    overlap = len(doc1_words & doc2_words)
                    union = len(doc1_words | doc2_words)
                    similarity = overlap / union if union > 0 else 0.0
                
                similarities.append(similarity)
        
        avg_similarity = np.mean(similarities) if similarities else 0.0
        return float(1.0 - avg_similarity)  # Higher diversity = lower similarity

class RetrievalBenchmark:
    """Benchmark retrieval systems with test queries."""
    
    def __init__(self):
        self.evaluator = RetrievalEvaluator()
        self.test_queries = self._create_test_queries()
    
    def _create_test_queries(self) -> List[Dict[str, Any]]:
        """Create test queries for compliance domain."""
        return [
            {
                "query": "What are the fundamental compliance requirements according to international standards?",
                "expected_topics": ["international standards", "fundamental requirements", "compliance"],
                "difficulty": "easy",
                "reference_answer": "International compliance standards establish fundamental requirements including adherence to legal frameworks, regulatory obligations, and ethical standards; transparency and accountability in operations; data protection and privacy safeguards; anti-corruption measures; environmental protection; and workplace safety standards. These requirements apply to organizations regardless of size, sector, or jurisdiction and must be integrated into operational procedures and governance structures."
            },
            {
                "query": "What are the main principles of GDPR compliance?",
                "expected_topics": ["GDPR", "data protection", "privacy", "compliance principles"],
                "difficulty": "medium",
                "reference_answer": "The main principles of GDPR compliance include lawfulness, fairness and transparency; purpose limitation; data minimization; accuracy; storage limitation; integrity and confidentiality; and accountability. Organizations must ensure they have a lawful basis for processing personal data, collect only necessary data for specified purposes, keep data accurate and up-to-date, implement appropriate security measures, and demonstrate compliance with all GDPR requirements."
            },
            {
                "query": "What compliance measures are required for AI systems?",
                "expected_topics": ["AI systems", "compliance", "regulations", "requirements"],
                "difficulty": "medium",
                "reference_answer": "Compliance measures required for AI systems include risk assessment and management frameworks, transparency and explainability requirements, data governance and privacy protections, bias and fairness testing, human oversight mechanisms, documentation of development processes, regular auditing and testing, security safeguards, and adherence to sector-specific regulations. Organizations must implement these measures throughout the AI lifecycle from design to deployment and ongoing monitoring, with particular attention to high-risk AI applications that may impact fundamental rights, safety, or critical infrastructure."
            },
            {
                "query": "How do European compliance frameworks differ from international standards?",
                "expected_topics": ["european frameworks", "international standards", "differences", "comparison"],
                "difficulty": "hard",
                "reference_answer": "European compliance frameworks and international standards differ in several key ways. European frameworks apply only to European states and organizations operating within the EU, with supranational enforcement mechanisms and binding decisions, while international standards are global with various enforcement mechanisms. European frameworks often have more stringent requirements and faster implementation timelines, while international standards provide broader flexibility for implementation. European frameworks allow direct enforcement actions, whereas international standards often rely on voluntary compliance and peer review mechanisms."
            },
            {
                "query": "What remedies are available for compliance violations?",
                "expected_topics": ["remedies", "violations", "legal recourse", "compensation"],
                "difficulty": "medium",
                "reference_answer": "Remedies for compliance violations include domestic judicial remedies through national courts, administrative remedies through regulatory agencies, international remedies through regional compliance bodies or international organizations, reparations including restitution, compensation, rehabilitation, satisfaction, and guarantees of non-repetition. Organizations may face monetary penalties, declaratory judgments, injunctive relief, criminal prosecution of responsible parties, and systemic reforms. The choice of remedy depends on the nature of the violation, available legal frameworks, and the jurisdiction where the violation occurred."
            }
        ]
    
    def run_benchmark(self, retrieval_function, k: int = 5) -> Dict[str, Any]:
        """Run benchmark evaluation on a retrieval function."""
        results = {
            "total_queries": len(self.test_queries),
            "k": k,
            "query_results": [],
            "aggregate_metrics": {}
        }
        
        all_precisions = []
        all_recalls = []
        all_f1s = []
        all_scores = []
        
        for test_query in self.test_queries:
            query = test_query["query"]
            
            try:
                # Get retrieval results
                retrieved_docs, scores = retrieval_function(query, k)
                
                # Evaluate
                metrics = self.evaluator.evaluate_retrieval_quality(
                    query, retrieved_docs, retrieval_scores=scores
                )
                
                # Add test-specific metrics
                metrics.update({
                    "expected_topics": test_query["expected_topics"],
                    "difficulty": test_query["difficulty"],
                    "topic_coverage": self._calculate_topic_coverage(
                        retrieved_docs, test_query["expected_topics"]
                    )
                })
                
                results["query_results"].append(metrics)
                
                # Collect for aggregation
                if scores:
                    all_scores.extend(scores)
                
            except Exception as e:
                logger.error(f"Error evaluating query '{query}': {e}")
                results["query_results"].append({
                    "query": query,
                    "error": str(e)
                })
        
        # Calculate aggregate metrics
        valid_results = [r for r in results["query_results"] if "error" not in r]
        
        if valid_results:
            results["aggregate_metrics"] = {
                "avg_num_retrieved": np.mean([r["num_retrieved"] for r in valid_results]),
                "avg_score": np.mean(all_scores) if all_scores else 0.0,
                "avg_diversity": np.mean([r.get("diversity_score", 0) for r in valid_results]),
                "success_rate": len(valid_results) / len(self.test_queries)
            }
            
            # Calculate average topic coverage by difficulty
            for difficulty in ["easy", "medium", "hard"]:
                difficulty_results = [r for r in valid_results if r.get("difficulty") == difficulty]
                if difficulty_results:
                    avg_coverage = np.mean([r.get("topic_coverage", 0) for r in difficulty_results])
                    results["aggregate_metrics"][f"avg_topic_coverage_{difficulty}"] = avg_coverage
        
        return results
    
    def _calculate_topic_coverage(self, documents: List[Document], 
                                 expected_topics: List[str]) -> float:
        """Calculate how well retrieved documents cover expected topics."""
        if not documents or not expected_topics:
            return 0.0
        
        # Combine all document content
        all_content = " ".join([doc.page_content.lower() for doc in documents])
        
        # Check topic coverage
        topics_found = 0
        for topic in expected_topics:
            if topic.lower() in all_content:
                topics_found += 1
        
        return topics_found / len(expected_topics)
