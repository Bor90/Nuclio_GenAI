"""
Text Generation Evaluator Library
This module provides evaluation functions for generated text output
from different decoding strategies.
"""

import re
from collections import Counter
from typing import Dict, Tuple, List
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class TextEvaluator:
    """Evaluator for text generation outputs"""
    
    def __init__(self):
        """Initialize the evaluator"""
        self.stop_words = set(stopwords.words('english'))
    
    def evaluate(self, generated_text: str, original_prompt: str = None) -> Dict:
        """
        Comprehensive evaluation of generated text
        
        Args:
            generated_text: The text to evaluate
            original_prompt: Optional original prompt for context
            
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['length'] = self._calculate_length(generated_text)
        metrics['repetition_score'] = self._calculate_repetition(generated_text)
        metrics['diversity_score'] = self._calculate_diversity(generated_text)
        metrics['coherence_score'] = self._calculate_coherence(generated_text)
        metrics['fluency_score'] = self._calculate_fluency(generated_text)
        
        return metrics
    
    def _calculate_length(self, text: str) -> Dict:
        """Calculate text length metrics"""
        words = word_tokenize(text)
        sentences = sent_tokenize(text)
        
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_words_per_sentence': len(words) / len(sentences) if sentences else 0,
            'char_count': len(text)
        }
    
    def _calculate_repetition(self, text: str) -> float:
        """
        Calculate repetition score (lower is better)
        Measures how many repeated n-grams appear in the text
        
        Range: 0-1 (0 = no repetition, 1 = high repetition)
        """
        words = word_tokenize(text.lower())
        
        if len(words) < 2:
            return 0.0
        
        # Calculate bigram repetition
        bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
        bigram_counts = Counter(bigrams)
        
        # Calculate trigram repetition
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)] if len(words) > 2 else []
        trigram_counts = Counter(trigrams)
        
        # Score based on how many bigrams/trigrams repeat
        repeated_bigrams = sum(1 for count in bigram_counts.values() if count > 1)
        repeated_trigrams = sum(1 for count in trigram_counts.values() if count > 1)
        
        total_repeated = repeated_bigrams + repeated_trigrams * 2
        total_ngrams = len(bigrams) + len(trigrams)
        
        repetition_score = min(1.0, total_repeated / total_ngrams) if total_ngrams > 0 else 0.0
        
        return round(repetition_score, 4)
    
    def _calculate_diversity(self, text: str) -> float:
        """
        Calculate diversity score (higher is better)
        Measures vocabulary richness (type-token ratio)
        
        Range: 0-1 (0 = no diversity, 1 = maximum diversity)
        """
        words = word_tokenize(text.lower())
        
        if len(words) == 0:
            return 0.0
        
        unique_words = len(set(words))
        diversity_score = unique_words / len(words)
        
        return round(diversity_score, 4)
    
    def _calculate_coherence(self, text: str) -> float:
        """
        Calculate coherence score (higher is better)
        Measures sentence-level structure and length consistency
        
        Range: 0-1
        """
        sentences = sent_tokenize(text)
        
        if len(sentences) < 2:
            return 1.0 if sentences else 0.0
        
        # Calculate average sentence length
        sentence_lengths = [len(word_tokenize(sent)) for sent in sentences]
        avg_length = sum(sentence_lengths) / len(sentence_lengths)
        
        # Penalize extremely short or inconsistent sentences
        inconsistency = sum(abs(length - avg_length) for length in sentence_lengths) / len(sentences)
        
        # Normalize inconsistency to 0-1 range
        coherence_score = 1.0 - min(1.0, inconsistency / (avg_length + 1))
        
        return round(coherence_score, 4)
    
    def _calculate_fluency(self, text: str) -> float:
        """
        Calculate fluency score (higher is better)
        Measures presence of common linguistic patterns
        
        Range: 0-1
        """
        words = word_tokenize(text.lower())
        
        if len(words) < 3:
            return 0.5
        
        # Count common stopwords (indicator of natural language)
        stopword_count = sum(1 for word in words if word in self.stop_words)
        
        # Calculate fluency based on stopword distribution
        # Natural text should have 30-50% stopwords
        stopword_ratio = stopword_count / len(words) if words else 0
        
        # Score is highest at 0.3-0.5 stopword ratio
        if 0.3 <= stopword_ratio <= 0.5:
            fluency_score = 1.0
        elif 0.2 <= stopword_ratio <= 0.6:
            fluency_score = 0.8
        elif 0.1 <= stopword_ratio <= 0.7:
            fluency_score = 0.6
        else:
            fluency_score = 0.4
        
        return round(fluency_score, 4)
    
    def compare_strategies(self, results: Dict[str, str]) -> Dict:
        """
        Compare multiple generation strategies
        
        Args:
            results: Dictionary with strategy names as keys and generated text as values
                    e.g., {'greedy': 'text1', 'beam_search': 'text2', ...}
        
        Returns:
            Dictionary with comparative analysis
        """
        evaluations = {}
        
        for strategy_name, generated_text in results.items():
            evaluations[strategy_name] = self.evaluate(generated_text)
        
        # Calculate comparative metrics
        comparison = self._create_comparison(evaluations)
        
        return {
            'individual_metrics': evaluations,
            'comparison': comparison
        }
    
    def _create_comparison(self, evaluations: Dict) -> Dict:
        """Create comparison between strategies"""
        if not evaluations:
            return {}
        
        comparison = {
            'best_diversity': max(evaluations.items(), key=lambda x: x[1]['diversity_score']),
            'lowest_repetition': min(evaluations.items(), key=lambda x: x[1]['repetition_score']),
            'best_coherence': max(evaluations.items(), key=lambda x: x[1]['coherence_score']),
            'best_fluency': max(evaluations.items(), key=lambda x: x[1]['fluency_score']),
        }
        
        return comparison
    
    def print_evaluation(self, metrics: Dict, strategy_name: str = None) -> None:
        """Pretty print evaluation metrics"""
        print("=" * 60)
        if strategy_name:
            print(f"Evaluation Results for: {strategy_name}")
        else:
            print("Evaluation Results")
        print("=" * 60)
        
        # Length metrics
        print("\n📏 LENGTH METRICS:")
        for key, value in metrics['length'].items():
            print(f"  {key}: {value}")
        
        # Quality metrics
        print(f"\n✨ QUALITY METRICS:")
        print(f"  Diversity Score: {metrics['diversity_score']:.4f} (higher is better)")
        print(f"  Repetition Score: {metrics['repetition_score']:.4f} (lower is better)")
        print(f"  Coherence Score: {metrics['coherence_score']:.4f} (higher is better)")
        print(f"  Fluency Score: {metrics['fluency_score']:.4f} (higher is better)")
        
        # Overall score
        overall = (metrics['diversity_score'] + 
                  (1 - metrics['repetition_score']) + 
                  metrics['coherence_score'] + 
                  metrics['fluency_score']) / 4
        
        print(f"\n🎯 OVERALL SCORE: {overall:.4f}/1.0")
        print("=" * 60)
    
    def print_comparison(self, comparison_results: Dict) -> None:
        """Pretty print comparison of strategies"""
        print("\n" + "=" * 60)
        print("STRATEGY COMPARISON SUMMARY")
        print("=" * 60)
        
        individual = comparison_results['individual_metrics']
        comparison = comparison_results['comparison']
        
        # Print individual scores
        print("\n📊 Individual Scores by Strategy:")
        print("-" * 60)
        
        for strategy, metrics in individual.items():
            overall = (metrics['diversity_score'] + 
                      (1 - metrics['repetition_score']) + 
                      metrics['coherence_score'] + 
                      metrics['fluency_score']) / 4
            
            print(f"\n{strategy.upper()}:")
            print(f"  Overall Score: {overall:.4f}")
            print(f"  Diversity: {metrics['diversity_score']:.4f}")
            print(f"  Repetition: {metrics['repetition_score']:.4f}")
            print(f"  Coherence: {metrics['coherence_score']:.4f}")
            print(f"  Fluency: {metrics['fluency_score']:.4f}")
        
        # Print winners
        print("\n🏆 Winners by Category:")
        print("-" * 60)
        for category, winner_info in comparison.items():
            strategy_name, _ = winner_info
            print(f"  {category.replace('_', ' ').title()}: {strategy_name}")
        
        print("\n" + "=" * 60)


def evaluate_text(text: str, original_prompt: str = None) -> Dict:
    """
    Simple function to evaluate a single text
    
    Args:
        text: Text to evaluate
        original_prompt: Optional original prompt
        
    Returns:
        Dictionary with metrics
    """
    evaluator = TextEvaluator()
    return evaluator.evaluate(text, original_prompt)


def compare_texts(results: Dict[str, str]) -> Dict:
    """
    Simple function to compare multiple texts
    
    Args:
        results: Dictionary with labels and texts
        
    Returns:
        Comparison results
    """
    evaluator = TextEvaluator()
    return evaluator.compare_strategies(results)


def print_results(metrics: Dict, strategy_name: str = None) -> None:
    """Pretty print results"""
    evaluator = TextEvaluator()
    evaluator.print_evaluation(metrics, strategy_name)


def print_comparison(comparison_results: Dict) -> None:
    """Pretty print comparison"""
    evaluator = TextEvaluator()
    evaluator.print_comparison(comparison_results)
