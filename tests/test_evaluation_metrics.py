"""
Unit tests for evaluation metrics module.

Tests Token F1, perplexity, and overall evaluation pipeline.
"""

import pytest
import torch
from src.evaluation.metrics import (
    normalize_answer,
    compute_token_f1,
    compute_perplexity,
    compute_evaluation_metrics
)


class TestNormalizeAnswer:
    """Test answer normalization"""

    def test_lowercase(self):
        assert normalize_answer("The Answer") == "answer"

    def test_remove_punctuation(self):
        assert normalize_answer("answer!") == "answer"
        assert normalize_answer("answer, here.") == "answer  here"

    def test_remove_articles(self):
        assert normalize_answer("the answer") == "answer"
        assert normalize_answer("an answer") == "answer"
        assert normalize_answer("a test") == "test"

    def test_remove_whitespace(self):
        assert normalize_answer("  answer  ") == "answer"
        assert normalize_answer("answer   here") == "answer here"

    def test_combined(self):
        assert normalize_answer("The Answer is: 42!") == "answer is 42"


class TestTokenF1:
    """Test Token F1 computation"""

    def test_perfect_match(self):
        pred = "the capital of france is paris"
        gold = "the capital of france is paris"
        assert compute_token_f1(pred, gold) == 1.0

    def test_perfect_match_after_normalization(self):
        pred = "The Capital of France is Paris!"
        gold = "the capital of france is paris"
        assert compute_token_f1(pred, gold) == 1.0

    def test_partial_overlap(self):
        pred = "paris is the capital"
        gold = "the capital is paris"
        # All 4 tokens (paris, is, capital) overlap
        # Precision = 3/4, Recall = 3/4, F1 = 3/4
        f1 = compute_token_f1(pred, gold)
        assert f1 == pytest.approx(0.75, abs=0.01)

    def test_no_overlap(self):
        pred = "london"
        gold = "paris"
        assert compute_token_f1(pred, gold) == 0.0

    def test_empty_prediction(self):
        pred = ""
        gold = "paris"
        assert compute_token_f1(pred, gold) == 0.0

    def test_empty_gold(self):
        pred = "paris"
        gold = ""
        assert compute_token_f1(pred, gold) == 0.0

    def test_both_empty(self):
        pred = ""
        gold = ""
        assert compute_token_f1(pred, gold) == 1.0

    def test_subset(self):
        pred = "paris"
        gold = "paris france"
        # Common: {paris}, Pred: {paris}, Gold: {paris, france}
        # Precision = 1/1 = 1.0, Recall = 1/2 = 0.5
        # F1 = 2 * (1.0 * 0.5) / (1.0 + 0.5) = 1.0 / 1.5 = 0.667
        f1 = compute_token_f1(pred, gold)
        assert f1 == pytest.approx(0.667, abs=0.01)

    def test_superset(self):
        pred = "paris france europe"
        gold = "paris"
        # Common: {paris}, Pred: {paris, france, europe}, Gold: {paris}
        # Precision = 1/3 = 0.333, Recall = 1/1 = 1.0
        # F1 = 2 * (0.333 * 1.0) / (0.333 + 1.0) = 0.666 / 1.333 = 0.5
        f1 = compute_token_f1(pred, gold)
        assert f1 == pytest.approx(0.5, abs=0.01)


class TestPerplexity:
    """Test perplexity computation (requires model)"""

    @pytest.mark.skip(reason="Requires loading model - run manually if needed")
    def test_perplexity_basic(self):
        """Basic test for perplexity (requires model loading)"""
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # Load small model for testing
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        text = "The capital of France is Paris."
        ppl = compute_perplexity(text, model, tokenizer, device="cpu")

        assert ppl > 0
        assert ppl < 1000  # Should be reasonable

    def test_perplexity_empty_text(self):
        """Test perplexity with empty text"""
        # Mock model and tokenizer (won't actually be called)
        model = None
        tokenizer = None

        ppl = compute_perplexity("", model, tokenizer, device="cpu")
        assert ppl == float('inf')


class TestEvaluationMetrics:
    """Test the combined evaluation metrics function"""

    @pytest.mark.skip(reason="Requires loading model - run manually if needed")
    def test_compute_evaluation_metrics(self):
        """Full integration test (requires model loading)"""
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # Load small model
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        response = "The capital of France is Paris"
        gold_answer = "Paris"

        metrics = compute_evaluation_metrics(
            response=response,
            gold_answer=gold_answer,
            model=model,
            tokenizer=tokenizer,
            device="cpu"
        )

        # Check all metrics are present
        assert 'f1' in metrics
        assert 'perplexity' in metrics
        assert 'response_length' in metrics

        # Check reasonable values
        assert 0.0 <= metrics['f1'] <= 1.0
        assert metrics['perplexity'] > 0
        assert metrics['response_length'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
