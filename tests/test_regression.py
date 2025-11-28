import pytest
import sys
import os
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantic_llm_compressor.compression.pipeline import CompressionPipeline
from semantic_llm_compressor.eval.benchmark import ModelBenchmark

class TestRegression:
    @pytest.fixture(scope="class")
    def compressed_model(self):
        """
        Creates a quick compressed model for testing if it doesn't exist.
        """
        model_path = "compressed_model_v2_test"
        if not os.path.exists(model_path):
            pipeline = CompressionPipeline(model_id="EleutherAI/gpt-neo-1.3B", device="cpu")
            # Use a very minimal config for speed, but enough to test the mechanism
            # We compress just ONE layer to be fast, but ideally we'd use the full v2 config
            # For regression, we want to match v2 logic.
            compression_config = {
                "target_modules": ["attn.attention.k_proj"],
                "rank": 32, # Low rank for speed in test
                "quantize": True,
                "exclude_layers": [4]
            }
            pipeline.compress(compression_config)
            pipeline.save_model(model_path)
        return model_path

    def test_compression_quality_sanity(self, compressed_model):
        """
        Verifies that the compressed model is not broken (Cosine > 0.95 for this quick test).
        """
        benchmark = ModelBenchmark(model_path=compressed_model, device="cpu")
        results = benchmark.run_full_suite(max_new_tokens=10, num_runs=1)
        
        cosine = results["quality"][0]["cosine_similarity"]
        print(f"Test Cosine Similarity: {cosine}")
        
        assert cosine > 0.95, f"Quality dropped too much! Cosine: {cosine}"

    def test_no_repetition_loop(self, compressed_model):
        """
        Checks if the model generates repetitive text (the Layer 4 bug).
        """
        benchmark = ModelBenchmark(model_path=compressed_model, device="cpu")
        prompt = "The meaning of life is"
        generated = benchmark.generate_text(prompt, max_new_tokens=30)
        
        print(f"Generated: {generated}")
        
        # Simple heuristic: check if a phrase repeats 3 times
        words = generated.split()
        ngrams = [tuple(words[i:i+4]) for i in range(len(words)-4)]
        if len(ngrams) > 0:
            most_common = max(set(ngrams), key=ngrams.count)
            count = ngrams.count(most_common)
            assert count < 3, f"Detected repetition loop! Phrase '{most_common}' repeated {count} times."

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
