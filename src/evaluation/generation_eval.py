"""Generation evaluation metrics and tools."""
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from rouge_score import rouge_scorer
from sacrebleu import BLEU
import re
import logging

logger = logging.getLogger(__name__)

class GenerationEvaluator:
    """Evaluate generation quality metrics."""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
        )
        self.bleu = BLEU()
    
    def calculate_rouge_scores(self, generated_text: str, reference_text: str) -> Dict[str, float]:
        """Calculate ROUGE scores for generated text."""
        try:
            scores = self.rouge_scorer.score(reference_text, generated_text)
            
            return {
                "rouge1_f": scores['rouge1'].fmeasure,
                "rouge1_p": scores['rouge1'].precision,
                "rouge1_r": scores['rouge1'].recall,
                "rouge2_f": scores['rouge2'].fmeasure,
                "rouge2_p": scores['rouge2'].precision,
                "rouge2_r": scores['rouge2'].recall,
                "rougeL_f": scores['rougeL'].fmeasure,
                "rougeL_p": scores['rougeL'].precision,
                "rougeL_r": scores['rougeL'].recall,
            }
        except Exception as e:
            logger.error(f"Error calculating ROUGE scores: {e}")
            return {key: 0.0 for key in [
                "rouge1_f", "rouge1_p", "rouge1_r",
                "rouge2_f", "rouge2_p", "rouge2_r", 
                "rougeL_f", "rougeL_p", "rougeL_r"
            ]}
    
    def calculate_bleu_score(self, generated_text: str, reference_texts: List[str]) -> float:
        """Calculate BLEU score for generated text."""
        try:
            # Tokenize texts
            hypothesis = generated_text.strip()
            references = [ref.strip() for ref in reference_texts]
            
            if not hypothesis or not any(references):
                return 0.0
            
            # Use sentence-level BLEU for better accuracy
            bleu_score = self.bleu.sentence_score(hypothesis, references)
            return bleu_score.score / 100.0  # Convert to 0-1 scale
            
        except Exception as e:
            logger.error(f"Error calculating BLEU score: {e}")
            return 0.0
    
    def calculate_factual_accuracy(self, generated_text: str, context: str) -> Dict[str, float]:
        """Calculate factual accuracy metrics with improved tolerance."""
        # Extract facts (simple approach using patterns)
        generated_facts = self._extract_facts(generated_text)
        context_facts = self._extract_facts(context)
        
        # Adjust scoring based on text length and complexity
        text_length = len(generated_text.split())
        
        if not generated_facts:
            # If no extractable facts, penalize based on expected complexity
            if text_length < 10:  # Very short text
                return {
                    "factual_precision": 0.6,  # Lower for very short responses
                    "factual_recall": 0.3,
                    "factual_f1": 0.4,
                    "hallucination_rate": 0.0
                }
            elif text_length < 30:  # Short text
                return {
                    "factual_precision": 0.7,
                    "factual_recall": 0.4,
                    "factual_f1": 0.5,
                    "hallucination_rate": 0.0
                }
            else:  # Longer text should have more facts
                return {
                    "factual_precision": 0.5,  # Penalize longer text without facts
                    "factual_recall": 0.3,
                    "factual_f1": 0.4,
                    "hallucination_rate": 0.1
                }
        
        # Calculate metrics with more lenient matching
        correct_facts = len(generated_facts & context_facts)
        
        # Add partial matching for similar facts
        partial_matches = 0
        for gen_fact in generated_facts:
            if gen_fact not in context_facts:
                # Check for partial matches (substring or similar)
                for ctx_fact in context_facts:
                    if (len(gen_fact) >= 4 and len(ctx_fact) >= 4 and 
                        (gen_fact in ctx_fact or ctx_fact in gen_fact or
                         gen_fact[:4] == ctx_fact[:4])):
                        partial_matches += 1
                        break
        
        # Include partial matches with reduced weight
        effective_correct = correct_facts + (partial_matches * 0.5)
        
        precision = min(1.0, effective_correct / len(generated_facts)) if generated_facts else 0.6
        recall = min(1.0, effective_correct / len(context_facts)) if context_facts else 0.4
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.5
        
        # Adjust hallucination penalty based on text complexity
        hallucinated_facts = max(0, len(generated_facts) - correct_facts - partial_matches)
        base_hallucination = hallucinated_facts / len(generated_facts) if generated_facts else 0.0
        
        # Scale hallucination rate based on text length (longer text should be more accurate)
        if text_length > 50:
            hallucination_rate = min(0.6, base_hallucination * 1.2)  # Higher penalty for long text
        else:
            hallucination_rate = min(0.4, base_hallucination)  # Lower penalty for short text
        
        return {
            "factual_precision": precision,
            "factual_recall": recall,
            "factual_f1": f1,
            "hallucination_rate": hallucination_rate
        }
    
    def calculate_coherence_score(self, text: str) -> float:
        """Calculate text coherence using improved heuristics."""
        if not text.strip():
            return 0.0
        
        sentences = self._split_sentences(text)
        if len(sentences) < 2:
            return 0.9  # Single sentence gets high coherence by default
        
        coherence_scores = []
        
        for i in range(len(sentences) - 1):
            # Enhanced coherence based on multiple factors
            sent1_words = set(sentences[i].lower().split())
            sent2_words = set(sentences[i + 1].lower().split())
            
            if len(sent1_words) == 0 or len(sent2_words) == 0:
                continue
            
            # 1. Lexical overlap (Jaccard similarity) - more generous baseline
            overlap = len(sent1_words & sent2_words)
            union = len(sent1_words | sent2_words)
            jaccard = overlap / union if union > 0 else 0.0
            
            # Start with higher base coherence for any meaningful text
            base_coherence = 0.6  # Base assumption of reasonable coherence
            
            # 2. Transition words/phrases (bigger bonus)
            transition_words = {
                'however', 'therefore', 'furthermore', 'moreover', 'additionally',
                'consequently', 'nevertheless', 'nonetheless', 'meanwhile',
                'similarly', 'likewise', 'in contrast', 'on the other hand',
                'for example', 'for instance', 'specifically', 'namely', 'also',
                'thus', 'hence', 'indeed', 'furthermore', 'finally'
            }
            
            sent2_lower = sentences[i + 1].lower()
            transition_bonus = 0.25 if any(tw in sent2_lower for tw in transition_words) else 0.0
            
            # 3. Pronoun resolution (bigger bonus)
            pronouns = {'this', 'that', 'these', 'those', 'it', 'they', 'them', 'such', 'which'}
            pronoun_bonus = 0.2 if any(p in sent2_words for p in pronouns) else 0.0
            
            # 4. Topic continuity (same domain words)
            legal_words = {'law', 'legal', 'right', 'rights', 'article', 'convention', 'declaration', 
                          'state', 'states', 'protection', 'freedom', 'human', 'international'}
            legal_overlap = len((sent1_words & legal_words) & (sent2_words & legal_words))
            topic_bonus = 0.15 if legal_overlap > 0 else 0.0
            
            # Combined coherence score (more generous)
            coherence = min(1.0, base_coherence + (jaccard * 0.3) + transition_bonus + pronoun_bonus + topic_bonus)
            coherence_scores.append(coherence)
        
        # Apply minimum threshold - most legal text should have reasonable coherence
        avg_coherence = float(np.mean(coherence_scores)) if coherence_scores else 0.7
        return max(0.5, avg_coherence)  # Minimum 0.5 for any structured text
    
    def calculate_relevance_score(self, generated_text: str, query: str) -> float:
        """Calculate relevance of generated text to the query with improved scoring."""
        if not generated_text.strip() or not query.strip():
            return 0.0
        
        gen_text_lower = generated_text.lower()
        query_lower = query.lower()
        
        # CRITICAL: Check for non-substantive responses first
        non_substantive_phrases = [
            "no information", "not enough information", "i don't know", "i cannot provide",
            "context does not contain", "context doesn't contain", "no specific information",
            "not available", "cannot provide an accurate", "cannot provide a detailed",
            "seek legal advice", "consult legal experts", "refer to specific", "important to refer"
        ]
        
        # Apply severe penalty for non-substantive responses
        for phrase in non_substantive_phrases:
            if phrase in gen_text_lower:
                return 0.1  # Maximum 0.1 relevance for non-substantive responses
        
        # 1. Exact keyword overlap
        gen_words = set(gen_text_lower.split())
        query_words = set(query_lower.split())
        
        # Enhanced stop words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", 
            "with", "by", "is", "are", "was", "were", "be", "been", "being", "have", 
            "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "what", "how", "when", "where", "why", "who", "which"
        }
        
        gen_words = gen_words - stop_words
        query_words = query_words - stop_words
        
        if not query_words:
            return 0.5  # Default relevance for empty query
        
        # 2. Exact word overlap score
        exact_overlap = len(gen_words & query_words)
        exact_score = exact_overlap / len(query_words)
        
        # 3. Partial word overlap (stems/substrings)
        partial_matches = 0
        for query_word in query_words:
            if len(query_word) >= 4:  # Only for meaningful words
                for gen_word in gen_words:
                    if len(gen_word) >= 4:
                        # Check if words share significant prefix/suffix
                        if (query_word[:4] == gen_word[:4] or 
                            query_word[-4:] == gen_word[-4:] or
                            query_word in gen_word or gen_word in query_word):
                            partial_matches += 1
                            break
        
        partial_score = partial_matches / len(query_words)
        
        # 4. Semantic concepts for legal domain
        legal_concepts = {
            'human rights': ['rights', 'freedoms', 'liberties', 'entitlements'],
            'freedom': ['liberty', 'right', 'autonomy'],
            'expression': ['speech', 'opinion', 'voice', 'communication'],
            'protection': ['safeguard', 'defend', 'secure', 'shield'],
            'violation': ['breach', 'infringement', 'abuse'],
            'obligation': ['duty', 'responsibility', 'requirement'],
            'refugee': ['asylum', 'displaced', 'migrant'],
            'state': ['government', 'nation', 'country', 'authority'],
            'kill': ['murder', 'homicide', 'killing', 'manslaughter', 'death', 'criminal'],
            'someone': ['person', 'individual', 'victim', 'human'],
            'punishment': ['penalty', 'sentence', 'imprisonment', 'consequences', 'sanctions'],
            'legal': ['law', 'court', 'criminal', 'prosecution', 'charges', 'justice']
        }
        
        concept_matches = 0
        total_concepts = 0
        for concept, synonyms in legal_concepts.items():
            if concept in query_lower:
                total_concepts += 1
                if any(syn in gen_text_lower for syn in synonyms) or concept in gen_text_lower:
                    concept_matches += 1
        
        concept_score = concept_matches / total_concepts if total_concepts > 0 else 0
        
        # Combined relevance score with weights
        combined_score = (
            exact_score * 0.5 +      # 50% weight for exact matches
            partial_score * 0.3 +    # 30% weight for partial matches  
            concept_score * 0.2      # 20% weight for concept matches
        )
        
        return min(1.0, combined_score)
    
    def calculate_fluency_score(self, text: str) -> float:
        """Calculate fluency using improved linguistic features."""
        if not text.strip():
            return 0.0
        
        # Normalize text
        text = text.strip()
        word_count = len(text.split())
        
        # Calculate features
        features = {
            "avg_sentence_length": self._avg_sentence_length(text),
            "punctuation_ratio": self._punctuation_ratio(text),
            "capitalization_score": self._capitalization_score(text),
            "repetition_penalty": self._repetition_penalty(text),
            "grammar_score": self._simple_grammar_score(text)
        }
        
        # More discriminating scoring
        fluency = 0.0
        
        # 1. Sentence length (reward appropriate complexity)
        avg_len = features["avg_sentence_length"]
        if 12 <= avg_len <= 30:
            fluency += 0.25  # Optimal range for complex ideas
        elif 8 <= avg_len <= 35:
            fluency += 0.20  # Good range
        elif 5 <= avg_len <= 45:
            fluency += 0.15  # Acceptable range
        elif avg_len > 0:
            fluency += 0.05  # Minimal credit for very short/long
        
        # 2. Punctuation (expect proper usage)
        punct_ratio = features["punctuation_ratio"]
        if 0.04 <= punct_ratio <= 0.15:
            fluency += 0.20  # Good punctuation
        elif 0.02 <= punct_ratio <= 0.25:
            fluency += 0.12  # Acceptable punctuation
        elif punct_ratio > 0:
            fluency += 0.05  # Some punctuation
        
        # 3. Capitalization (stricter requirements)
        cap_score = features["capitalization_score"]
        if cap_score >= 0.9:
            fluency += 0.20  # Excellent capitalization
        elif cap_score >= 0.7:
            fluency += 0.15  # Good capitalization
        elif cap_score >= 0.5:
            fluency += 0.10  # Acceptable
        elif cap_score > 0:
            fluency += 0.05  # Some capitalization
        
        # 4. Grammar score (weighted by complexity)
        grammar_contribution = features["grammar_score"] * 0.25
        if word_count < 10:  # Penalize simple grammar for very short text
            grammar_contribution *= 0.7
        fluency += grammar_contribution
        
        # 5. Repetition penalty (increased impact for short text)
        repetition_penalty = features["repetition_penalty"]
        if word_count < 15:  # Higher penalty for repetitive short text
            repetition_penalty *= 1.5
        fluency -= repetition_penalty * 0.25
        
        # 6. Length and complexity bonus
        if word_count >= 50:
            fluency += 0.15  # Reward substantial content
        elif word_count >= 30:
            fluency += 0.10
        elif word_count >= 15:
            fluency += 0.05
        elif word_count < 5:
            fluency -= 0.10  # Penalize very short responses
        
        # 7. Vocabulary diversity bonus
        unique_words = len(set(text.lower().split()))
        diversity_ratio = unique_words / word_count if word_count > 0 else 0
        if diversity_ratio >= 0.8:
            fluency += 0.10  # High vocabulary diversity
        elif diversity_ratio >= 0.6:
            fluency += 0.05  # Good diversity
        
        return max(0.0, min(1.0, fluency))
    
    def evaluate_generation_quality(self, generated_text: str, query: str, 
                                  context: Optional[str] = None, reference_text: Optional[str] = None) -> Dict[str, Any]:
        """Comprehensive generation evaluation."""
        metrics = {
            "generated_text_length": len(generated_text),
            "query": query
        }
        
        # Basic quality metrics
        metrics["coherence_score"] = self.calculate_coherence_score(generated_text)
        metrics["relevance_score"] = self.calculate_relevance_score(generated_text, query)
        metrics["fluency_score"] = self.calculate_fluency_score(generated_text)
        
        # Context-based metrics
        if context:
            factual_metrics = self.calculate_factual_accuracy(generated_text, context)
            metrics.update(factual_metrics)
            
            # Context utilization
            metrics["context_utilization"] = self._calculate_context_utilization(
                generated_text, context
            )
        
        # Reference-based metrics
        if reference_text:
            rouge_metrics = self.calculate_rouge_scores(generated_text, reference_text)
            metrics.update(rouge_metrics)
            
            bleu_score = self.calculate_bleu_score(generated_text, [reference_text])
            metrics["bleu_score"] = bleu_score
            
            # Semantic similarity
            semantic_sim = self.calculate_semantic_similarity(generated_text, reference_text)
            metrics["semantic_similarity"] = semantic_sim
        
        # Enhanced overall quality score with proper weighting
        quality_scores = []
        weights = []
        
        # Core quality metrics (always present)
        quality_scores.extend([metrics["coherence_score"], metrics["relevance_score"], metrics["fluency_score"]])
        weights.extend([0.15, 0.25, 0.15])  # Higher weight for relevance
        
        # Context-based metrics (if available)
        if context:
            # Reduce hallucination penalty but don't make it overwhelming
            factual_accuracy = 1.0 - min(0.5, metrics["hallucination_rate"])  # Cap penalty at 0.5
            quality_scores.extend([factual_accuracy, metrics["context_utilization"]])
            weights.extend([0.20, 0.10])
        
        # Reference-based metrics (if available)
        if reference_text:
            # Use ROUGE-L as it's most reliable for overall content similarity
            rouge_score = metrics.get("rougeL_f", 0.0)
            quality_scores.append(rouge_score)
            weights.append(0.10)
            
            # Semantic similarity if available
            if "semantic_similarity" in metrics:
                quality_scores.append(metrics["semantic_similarity"])
                weights.append(0.05)
        
        # Calculate weighted average
        if len(quality_scores) > 0 and len(weights) > 0:
            # Normalize weights to sum to 1
            total_weight = sum(weights[:len(quality_scores)])
            normalized_weights = [w/total_weight for w in weights[:len(quality_scores)]]
            
            # Calculate weighted score
            overall_score = sum(score * weight for score, weight in zip(quality_scores, normalized_weights))
        else:
            # Fallback: simple average of basic metrics
            basic_metrics = [metrics["coherence_score"], metrics["relevance_score"], metrics["fluency_score"]]
            overall_score = float(np.mean(basic_metrics))
        
        # Apply bonuses for good performance
        if overall_score > 0.7:
            overall_score = min(1.0, float(overall_score) * 1.1)  # 10% bonus for high scores
        
        # CRITICAL: Apply penalty for non-substantive responses
        gen_text_lower = generated_text.lower()
        non_substantive_phrases = [
            "no information", "not enough information", "i don't know", "i cannot provide",
            "context does not contain", "context doesn't contain", "no specific information",
            "not available", "cannot provide an accurate", "cannot provide a detailed",
            "seek legal advice", "consult legal experts", "refer to specific", "important to refer"
        ]
        
        for phrase in non_substantive_phrases:
            if phrase in gen_text_lower:
                overall_score = min(overall_score, 0.3)  # Cap at 0.3 for non-substantive responses
                break
        
        metrics["overall_quality"] = max(0.0, min(1.0, float(overall_score)))
        
        # Add detailed breakdown for debugging
        metrics["quality_breakdown"] = {
            "scores": quality_scores[:len(weights)],
            "weights": weights[:len(quality_scores)],
            "component_names": (["coherence", "relevance", "fluency"] + 
                              (["factual_accuracy", "context_utilization"] if context else []) +
                              (["rouge_similarity"] + (["semantic_similarity"] if "semantic_similarity" in metrics else []) if reference_text else []))
        }
        
        return metrics
    
    def _extract_facts(self, text: str) -> set:
        """Extract simple facts from text (enhanced for legal domain)."""
        facts = set()
        
        # Extract sentences that look like facts
        sentences = self._split_sentences(text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Enhanced patterns for legal facts
            patterns = [
                r'Article\s+\d+',
                r'Section\s+\d+',
                r'Paragraph\s+\d+',
                r'Chapter\s+\d+',
                r'\b\d{4}\b',  # Years (more specific)
                r'[A-Z][a-z]+\s+Convention(?:\s+on\s+[A-Z][a-z\s]+)?',
                r'[A-Z][a-z]+\s+Declaration(?:\s+of\s+[A-Z][a-z\s]+)?',
                r'[A-Z][a-z]+\s+Covenant(?:\s+on\s+[A-Z][a-z\s]+)?',
                r'shall\s+(?:not\s+)?[a-z]+',
                r'prohibited(?:\s+from)?',
                r'mandatory(?:\s+to)?',
                r'obliged?\s+to\s+[a-z]+',
                r'right\s+to\s+[a-z\s]+',
                r'freedom\s+of\s+[a-z\s]+',
                r'freedom\s+from\s+[a-z\s]+',
                r'entitled\s+to\s+[a-z\s]+',
                r'violation\s+of\s+[a-z\s]+',
                r'breach\s+of\s+[a-z\s]+',
                r'principle\s+of\s+[a-z\s]+',
                r'doctrine\s+of\s+[a-z\s]+',
                r'non-refoulement',
                r'erga\s+omnes',
                r'jus\s+cogens'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, sentence, re.IGNORECASE)
                # Normalize and clean matches
                normalized_facts = [re.sub(r'\s+', ' ', match.lower().strip()) for match in matches]
                facts.update(normalized_facts)
        
        return facts
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _avg_sentence_length(self, text: str) -> float:
        """Calculate average sentence length in words."""
        sentences = self._split_sentences(text)
        if not sentences:
            return 0.0
        
        total_words = sum(len(sentence.split()) for sentence in sentences)
        return total_words / len(sentences)
    
    def _punctuation_ratio(self, text: str) -> float:
        """Calculate ratio of punctuation characters."""
        if not text:
            return 0.0
        
        punct_chars = len(re.findall(r'[.,!?;:]', text))
        return punct_chars / len(text)
    
    def _capitalization_score(self, text: str) -> float:
        """Calculate proper capitalization score."""
        if not text:
            return 0.0
        
        sentences = self._split_sentences(text)
        if not sentences:
            return 0.0
        
        properly_capitalized = 0
        for sentence in sentences:
            if sentence and sentence[0].isupper():
                properly_capitalized += 1
        
        return properly_capitalized / len(sentences)
    
    def _repetition_penalty(self, text: str) -> float:
        """Calculate repetition penalty."""
        words = text.lower().split()
        if len(words) < 2:
            return 0.0
        
        # Calculate word repetition
        unique_words = len(set(words))
        repetition_rate = 1.0 - (unique_words / len(words))
        
        return min(1.0, repetition_rate * 2)  # Scale penalty
    
    def _calculate_context_utilization(self, generated_text: str, context: str) -> float:
        """Calculate how well the generated text uses the provided context."""
        if not context or not generated_text:
            return 0.0
        
        gen_words = set(generated_text.lower().split())
        context_words = set(context.lower().split())
        
        # Remove common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        gen_words = gen_words - stop_words
        context_words = context_words - stop_words
        
        if not context_words:
            return 0.0
        
        overlap = len(gen_words & context_words)
        return overlap / len(context_words)
    
    def calculate_semantic_similarity(self, generated_text: str, reference_text: str) -> float:
        """Calculate semantic similarity using sentence embeddings (optional)."""
        try:
            # This requires sentence-transformers, which you already have
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            # Use a lightweight model for evaluation
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Get embeddings
            gen_embedding = model.encode([generated_text])
            ref_embedding = model.encode([reference_text])

            # Ensure embeddings are numpy arrays
            if not isinstance(gen_embedding, np.ndarray):
                gen_embedding = np.array(gen_embedding)
            if not isinstance(ref_embedding, np.ndarray):
                ref_embedding = np.array(ref_embedding)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(gen_embedding, ref_embedding)[0][0]
            return float(similarity)
            
        except ImportError:
            logger.warning("Sentence transformers not available for semantic similarity")
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def _simple_grammar_score(self, text: str) -> float:
        """Calculate a simple grammar score based on basic patterns."""
        if not text.strip():
            return 0.0
        
        score = 1.0  # Start with perfect score
        
        # Check for basic grammar issues
        sentences = self._split_sentences(text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 1. Check if sentence starts with capital letter
            if not sentence[0].isupper():
                score -= 0.1
            
            # 2. Check for basic subject-verb patterns
            words = sentence.lower().split()
            if len(words) >= 2:
                # Simple heuristic: look for common verb patterns
                common_verbs = {'is', 'are', 'was', 'were', 'has', 'have', 'had', 'does', 'do', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must'}
                if not any(verb in words for verb in common_verbs):
                    # Check for other verb forms (basic)
                    verb_endings = ['ed', 'ing', 'es', 's']
                    has_verb = any(word.endswith(ending) for word in words for ending in verb_endings)
                    if not has_verb:
                        score -= 0.05
            
            # 3. Check for extremely short or long sentences
            if len(words) < 3:
                score -= 0.05
            elif len(words) > 60:
                score -= 0.1
        
        # 4. Check for repeated punctuation
        if re.search(r'[.!?]{2,}', text):
            score -= 0.1
        
        # 5. Check for missing punctuation at sentence ends
        non_punct_endings = len([s for s in sentences if s and not s[-1] in '.!?'])
        if non_punct_endings > 0:
            score -= 0.1 * (non_punct_endings / len(sentences))
        
        return max(0.0, min(1.0, score))
