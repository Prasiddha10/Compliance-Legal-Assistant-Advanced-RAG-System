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
        """Create test queries for human rights domain."""
        return [
            {
                "query": "What are the fundamental human rights according to the Universal Declaration?",
                "expected_topics": ["universal declaration", "fundamental rights", "human rights"],
                "difficulty": "easy",
                "reference_answer": "The Universal Declaration of Human Rights establishes fundamental human rights including the right to life, liberty, and security of person; freedom from slavery and torture; equality before the law; fair trial rights; freedom of thought, conscience, and religion; freedom of expression and assembly; and rights to education, work, and adequate standard of living. These rights are inherent to all human beings regardless of race, sex, nationality, ethnicity, language, religion, or other status."
            },
            {
                "query": "How does international law protect freedom of expression?",
                "expected_topics": ["freedom of expression", "international law", "protection"],
                "difficulty": "medium",
                "reference_answer": "International law protects freedom of expression through Article 19 of the Universal Declaration of Human Rights and Article 19 of the International Covenant on Civil and Political Rights. This protection includes the right to seek, receive, and impart information and ideas through any media regardless of frontiers. However, this right may be subject to certain restrictions for respect of the rights of others, protection of national security, public order, or public health and morals, but such restrictions must be provided by law and necessary in a democratic society."
            },
            {
                "query": "What are the obligations of states regarding refugee protection?",
                "expected_topics": ["refugee", "state obligations", "protection", "asylum"],
                "difficulty": "medium",
                "reference_answer": "States have several key obligations regarding refugee protection under international law, particularly the 1951 Refugee Convention and its 1967 Protocol. These include the principle of non-refoulement (not returning refugees to territories where they face threats), providing access to fair and efficient asylum procedures, ensuring refugees have access to basic rights including education, healthcare, and work, and cooperating with UNHCR in the protection of refugees. States must also ensure that refugee status determination is conducted fairly and that recognized refugees receive appropriate documentation."
            },
            {
                "query": "How does the European Convention on Human Rights differ from the ICCPR?",
                "expected_topics": ["european convention", "ICCPR", "differences", "comparison"],
                "difficulty": "hard",
                "reference_answer": "The European Convention on Human Rights (ECHR) and the International Covenant on Civil and Political Rights (ICCPR) differ in several key ways. The ECHR applies only to European states and has a supranational court system (European Court of Human Rights) with binding decisions, while the ICCPR is global with a Human Rights Committee that issues non-binding views. The ECHR covers primarily civil and political rights with some implied economic rights, while the ICCPR focuses exclusively on civil and political rights. The ECHR allows individual petitions directly to the court, whereas the ICCPR requires state ratification of the Optional Protocol for individual complaints."
            },
            {
                "query": "What remedies are available for human rights violations?",
                "expected_topics": ["remedies", "violations", "legal recourse", "compensation"],
                "difficulty": "medium",
                "reference_answer": "Remedies for human rights violations include domestic judicial remedies through national courts, administrative remedies through government agencies, international remedies through regional human rights courts or UN treaty bodies, reparations including restitution, compensation, rehabilitation, satisfaction, and guarantees of non-repetition. Victims may seek monetary damages, declaratory judgments, injunctive relief, criminal prosecution of perpetrators, and systemic reforms. The choice of remedy depends on the nature of the violation, available legal frameworks, and the jurisdiction where the violation occurred."
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
