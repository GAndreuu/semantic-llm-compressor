# scripts/profile_memory_and_latency.py

import argparse
import time
import psutil
import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from semantic_llm_compressor.runtime.patcher import patch_model_with_compressed_linears

def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def measure_inference(model, tokenizer, prompt, device, num_runs=5, max_new_tokens=20):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Warmup
    model.generate(**inputs, max_new_tokens=5, do_sample=False)
    
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        end = time.perf_counter()
        times.append(end - start)
        
    avg_time = np.mean(times)
    std_time = np.std(times)
    tokens_per_sec = max_new_tokens / avg_time
    
    return avg_time, std_time, tokens_per_sec

def main():
    parser = argparse.ArgumentParser(description="Profile memory and latency.")
    parser.add_argument("--original_dir", type=str, required=True)
    parser.add_argument("--compressed_dir", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompt = "The quick brown fox jumps over the lazy dog"
    
    print(f"Profiling on device: {device}")
    
    # --- Original Model ---
    print("\n--- Original Model ---")
    mem_before = get_memory_usage_mb()
    model_orig = AutoModelForCausalLM.from_pretrained(args.original_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.original_dir)
    mem_after = get_memory_usage_mb()
    
    print(f"Memory (Load): {mem_after - mem_before:.2f} MB (Approximation)")
    
    avg, std, tps = measure_inference(model_orig, tokenizer, prompt, device)
    print(f"Latency: {avg:.4f}s ± {std:.4f}s")
    print(f"Throughput: {tps:.2f} tokens/s")
    
    del model_orig
    torch.cuda.empty_cache()
    
    # --- Compressed Model ---
    print("\n--- Compressed Model ---")
    mem_before = get_memory_usage_mb()
    model_comp = AutoModelForCausalLM.from_pretrained(args.original_dir).to(device)
    patch_model_with_compressed_linears(model_comp, args.compressed_dir, verbose=False)
    mem_after = get_memory_usage_mb()
    
    print(f"Memory (Load + Patch): {mem_after - mem_before:.2f} MB (Approximation)")
    
    avg, std, tps = measure_inference(model_comp, tokenizer, prompt, device)
    print(f"Latency: {avg:.4f}s ± {std:.4f}s")
    print(f"Throughput: {tps:.2f} tokens/s")

if __name__ == "__main__":
    main()
