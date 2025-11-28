import argparse
import sys
import os
import json
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from semantic_llm_compressor.config import ProjectConfig
from semantic_llm_compressor.compression.pipeline import CompressionPipeline
from semantic_llm_compressor.eval.benchmark import ModelBenchmark

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Semantic LLM Compressor Pipeline")
    parser.add_argument("--action", choices=["all", "prepare", "compress", "benchmark"], default="all", help="Action to perform")
    parser.add_argument("--rank", type=int, default=128, help="SVD Rank")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark (fewer tokens/runs)")
    parser.add_argument("--model", type=str, default="EleutherAI/gpt-neo-1.3B", help="Model ID")
    
    args = parser.parse_args()
    
    config = ProjectConfig()
    
    # 1. Prepare (Implicitly handled by library loading, but we can verify)
    if args.action in ["all", "prepare"]:
        logger.info(f"Preparing model {args.model}...")
        # In a real scenario, we might download here. For now, we assume it's cached or handled by HF.
        pass

    # 2. Compress
    if args.action in ["all", "compress"]:
        logger.info(f"Starting compression with Rank {args.rank}...")
        pipeline = CompressionPipeline(model_id=args.model, device="cpu")
        
        # Define compression config (v2 strategy)
        compression_config = {
            "target_modules": ["attn.attention.k_proj"],
            "rank": args.rank,
            "quantize": True,
            "exclude_layers": [4] # Critical fix from v2
        }
        
        compressed_model = pipeline.compress(compression_config)
        pipeline.save_model("compressed_model_v2")
        logger.info("Compression complete. Model saved to 'compressed_model_v2'.")

    # 3. Benchmark
    if args.action in ["all", "benchmark"]:
        logger.info("Starting benchmark...")
        benchmark = ModelBenchmark(model_path="compressed_model_v2", device="cpu")
        
        max_new_tokens = 10 if args.quick else 50
        num_runs = 1 if args.quick else 5
        
        results = benchmark.run_full_suite(
            max_new_tokens=max_new_tokens,
            num_runs=num_runs
        )
        
        print(json.dumps(results, indent=2))
        
        # Quick quality check
        if results["quality"][0]["cosine_similarity"] < 0.99:
            logger.warning("⚠️ Quality Warning: Cosine Similarity is below 0.99!")
        else:
            logger.info("✅ Quality Check Passed")

if __name__ == "__main__":
    main()
