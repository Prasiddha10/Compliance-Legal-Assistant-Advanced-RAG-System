"""LLM as Judge implementation for evaluation."""
from typing import Dict, Any, List, Optional
from langchain.prompts import PromptTemplate
from src.rag.llm_manager import LLMManager
import json
import logging

logger = logging.getLogger(__name__)

class LLMJudge:
    """Use LLM as a judge for evaluating RAG responses."""
    
    def __init__(self, llm_manager: Optional[LLMManager] = None, judge_model: Optional[str] = None):
        self.llm_manager = llm_manager or LLMManager()
        self.judge_model = judge_model or "gpt-4"  # Use GPT-4 as default judge
        
        # Initialize judge prompts
        self.prompts = self._initialize_prompts()
        
        logger.info(f"LLM Judge initialized with model: {self.judge_model}")
    
    def _initialize_prompts(self) -> Dict[str, PromptTemplate]:
        """Initialize evaluation prompts."""
        
        # Overall quality evaluation
        quality_prompt = PromptTemplate(
            input_variables=["query", "context", "response"],
            template="""You are an expert evaluator for legal and human rights question-answering systems.
            Evaluate the following response based on the given query and context.
            
            Query: {query}
            
            Context: {context}
            
            Response: {response}
            
            Please evaluate the response on the following criteria (score 1-10):
            
            1. ACCURACY: How factually correct is the response based on the context?
            2. RELEVANCE: How well does the response address the specific query?
            3. COMPLETENESS: How thoroughly does the response answer the question?
            4. CLARITY: How clear and well-structured is the response?
            5. CONTEXT_USE: How well does the response utilize the provided context?
            
            Provide your evaluation in the following JSON format:
            {{
                "accuracy": <score>,
                "relevance": <score>,
                "completeness": <score>,
                "clarity": <score>,
                "context_use": <score>,
                "overall_score": <average_score>
            }}"""
        )
        
        # Factual accuracy evaluation
        factual_prompt = PromptTemplate(
            input_variables=["context", "response"],
            template="""You are an expert fact-checker for legal documents.
            Evaluate whether the following response contains any factual errors or hallucinations
            compared to the provided context.
            
            Context: {context}
            
            Response: {response}
            
            Check for:
            1. Claims made in the response that are not supported by the context
            2. Incorrect legal references or citations
            3. Misstatements of law or facts
            4. Hallucinated information
            
            Provide your evaluation in JSON format:
            {{
                "factual_accuracy_score": <score_1_to_10>,
                "has_hallucinations": <true/false>,
                "unsupported_claims": ["<claim1>", "<claim2>"],
                "incorrect_facts": ["<fact1>", "<fact2>"],
                "explanation": "<explanation>"
            }}"""
        )
        
        # Legal domain relevance
        legal_relevance_prompt = PromptTemplate(
            input_variables=["query", "response"],
            template="""You are an expert in human rights law and legal systems.
            Evaluate how well the following response addresses the legal query in terms of:
            
            Query: {query}
            Response: {response}
            
            Criteria:
            1. Legal accuracy and proper use of legal terminology
            2. Relevance to human rights law
            3. Appropriate legal reasoning
            4. Citation of relevant legal instruments or principles
            
            Provide evaluation in JSON format:
            {{
                "legal_accuracy": <score_1_to_10>,
                "human_rights_relevance": <score_1_to_10>,
                "legal_reasoning": <score_1_to_10>,
                "proper_citations": <score_1_to_10>,
                "overall_legal_quality": <average_score>,
                "legal_concepts_identified": ["<concept1>", "<concept2>"],
                "explanation": "<explanation>"
            }}"""
        )
        
        return {
            "quality": quality_prompt,
            "factual": factual_prompt,
            "legal": legal_relevance_prompt
        }
    
    def evaluate_response_quality(self, query: str, context: str, response: str) -> Dict[str, Any]:
        """Evaluate overall response quality using LLM judge."""
        try:
            judge_llm = self.llm_manager.get_model(self.judge_model)
            prompt = self.prompts["quality"]
            
            # Generate evaluation
            if hasattr(judge_llm, 'invoke'):
                evaluation_text = judge_llm.invoke(
                    prompt.format(query=query, context=context, response=response)
                )
                if hasattr(evaluation_text, 'content'):
                    evaluation_text = evaluation_text.content
            else:
                evaluation_text = judge_llm(
                    prompt.format(query=query, context=context, response=response)
                )
            
            # Parse JSON response
            try:
                evaluation = json.loads(evaluation_text)
            except json.JSONDecodeError:
                # If JSON parsing fails, extract scores manually
                evaluation = self._extract_scores_from_text(evaluation_text)
            
            evaluation["judge_model"] = self.judge_model
            evaluation["evaluation_type"] = "quality"
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error in quality evaluation: {e}")
            return {
                "error": str(e),
                "judge_model": self.judge_model,
                "evaluation_type": "quality"
            }
    
    def evaluate_factual_accuracy(self, context: str, response: str) -> Dict[str, Any]:
        """Evaluate factual accuracy using LLM judge."""
        try:
            judge_llm = self.llm_manager.get_model(self.judge_model)
            prompt = self.prompts["factual"]
            
            # Generate evaluation
            if hasattr(judge_llm, 'invoke'):
                evaluation_text = judge_llm.invoke(
                    prompt.format(context=context, response=response)
                )
                if hasattr(evaluation_text, 'content'):
                    evaluation_text = evaluation_text.content
            else:
                evaluation_text = judge_llm(
                    prompt.format(context=context, response=response)
                )
            
            # Parse JSON response
            try:
                evaluation = json.loads(evaluation_text)
            except json.JSONDecodeError:
                evaluation = self._extract_factual_scores_from_text(evaluation_text)
            
            evaluation["judge_model"] = self.judge_model
            evaluation["evaluation_type"] = "factual"
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error in factual evaluation: {e}")
            return {
                "error": str(e),
                "judge_model": self.judge_model,
                "evaluation_type": "factual"
            }
    
    def evaluate_legal_relevance(self, query: str, response: str) -> Dict[str, Any]:
        """Evaluate legal domain relevance using LLM judge."""
        try:
            judge_llm = self.llm_manager.get_model(self.judge_model)
            prompt = self.prompts["legal"]
            
            # Generate evaluation
            if hasattr(judge_llm, 'invoke'):
                evaluation_text = judge_llm.invoke(
                    prompt.format(query=query, response=response)
                )
                if hasattr(evaluation_text, 'content'):
                    evaluation_text = evaluation_text.content
            else:
                evaluation_text = judge_llm(
                    prompt.format(query=query, response=response)
                )
            
            # Parse JSON response
            try:
                evaluation = json.loads(evaluation_text)
            except json.JSONDecodeError:
                evaluation = self._extract_legal_scores_from_text(evaluation_text)
            
            evaluation["judge_model"] = self.judge_model
            evaluation["evaluation_type"] = "legal"
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error in legal evaluation: {e}")
            return {
                "error": str(e),
                "judge_model": self.judge_model,
                "evaluation_type": "legal"
            }
    
    def comprehensive_evaluation(self, query: str, context: str, response: str) -> Dict[str, Any]:
        """Run comprehensive evaluation using all judge criteria."""
        evaluations = {}
        
        # Quality evaluation
        quality_eval = self.evaluate_response_quality(query, context, response)
        evaluations["quality_evaluation"] = quality_eval
        
        # Factual accuracy evaluation
        factual_eval = self.evaluate_factual_accuracy(context, response)
        evaluations["factual_evaluation"] = factual_eval
        
        # Legal relevance evaluation
        legal_eval = self.evaluate_legal_relevance(query, response)
        evaluations["legal_evaluation"] = legal_eval
        
        # Calculate overall score
        scores = []
        if "overall_score" in quality_eval:
            scores.append(quality_eval["overall_score"])
        if "factual_accuracy_score" in factual_eval:
            scores.append(factual_eval["factual_accuracy_score"])
        if "overall_legal_quality" in legal_eval:
            scores.append(legal_eval["overall_legal_quality"])
        
        evaluations["comprehensive_score"] = sum(scores) / len(scores) if scores else 0.0
        evaluations["judge_model"] = self.judge_model
        
        return evaluations
    
    def _extract_scores_from_text(self, text: str) -> Dict[str, Any]:
        """Extract scores from text when JSON parsing fails."""
        import re
        
        # Default scores
        scores = {
            "accuracy": 5.0,
            "relevance": 5.0,
            "completeness": 5.0,
            "clarity": 5.0,
            "context_use": 5.0,
            "overall_score": 5.0,
            "explanation": "Could not parse detailed evaluation",
            "strengths": [],
            "weaknesses": []
        }
        
        # Try to extract numeric scores
        score_patterns = [
            (r'accuracy[:\s]+(\d+(?:\.\d+)?)', 'accuracy'),
            (r'relevance[:\s]+(\d+(?:\.\d+)?)', 'relevance'),
            (r'completeness[:\s]+(\d+(?:\.\d+)?)', 'completeness'),
            (r'clarity[:\s]+(\d+(?:\.\d+)?)', 'clarity'),
            (r'context_use[:\s]+(\d+(?:\.\d+)?)', 'context_use'),
            (r'overall[:\s]+(\d+(?:\.\d+)?)', 'overall_score')
        ]
        
        for pattern, key in score_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                scores[key] = float(match.group(1))
        
        # Calculate overall if not found
        if scores["overall_score"] == 5.0:
            component_scores = [scores[k] for k in ["accuracy", "relevance", "completeness", "clarity", "context_use"]]
            scores["overall_score"] = sum(component_scores) / len(component_scores)
        
        return scores
    
    def _extract_factual_scores_from_text(self, text: str) -> Dict[str, Any]:
        """Extract factual scores from text when JSON parsing fails."""
        import re
        
        scores = {
            "factual_accuracy_score": 5.0,
            "has_hallucinations": False,
            "unsupported_claims": [],
            "incorrect_facts": [],
            "explanation": "Could not parse detailed evaluation"
        }
        
        # Extract factual accuracy score
        score_match = re.search(r'factual_accuracy[:\s]+(\d+(?:\.\d+)?)', text, re.IGNORECASE)
        if score_match:
            scores["factual_accuracy_score"] = float(score_match.group(1))
        
        # Check for hallucinations
        if any(word in text.lower() for word in ["hallucination", "unsupported", "incorrect", "false"]):
            scores["has_hallucinations"] = True
        
        return scores
    
    def _extract_legal_scores_from_text(self, text: str) -> Dict[str, Any]:
        """Extract legal scores from text when JSON parsing fails."""
        import re
        
        scores = {
            "legal_accuracy": 5.0,
            "human_rights_relevance": 5.0,
            "legal_reasoning": 5.0,
            "proper_citations": 5.0,
            "overall_legal_quality": 5.0,
            "legal_concepts_identified": [],
            "explanation": "Could not parse detailed evaluation"
        }
        
        # Extract scores
        score_patterns = [
            (r'legal_accuracy[:\s]+(\d+(?:\.\d+)?)', 'legal_accuracy'),
            (r'human_rights_relevance[:\s]+(\d+(?:\.\d+)?)', 'human_rights_relevance'),
            (r'legal_reasoning[:\s]+(\d+(?:\.\d+)?)', 'legal_reasoning'),
            (r'proper_citations[:\s]+(\d+(?:\.\d+)?)', 'proper_citations')
        ]
        
        for pattern, key in score_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                scores[key] = float(match.group(1))
        
        # Calculate overall
        component_scores = [scores[k] for k in ["legal_accuracy", "human_rights_relevance", "legal_reasoning", "proper_citations"]]
        scores["overall_legal_quality"] = sum(component_scores) / len(component_scores)
        
        return scores
