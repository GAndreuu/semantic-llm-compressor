# scripts/5_benchmark_gpt_neo.py

import argparse
import time
import psutil
import os
import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from semantic_llm_compressor.runtime.patcher import patch_model_with_compressed_linears

def get_memory_usage_mb():
    """Get current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def measure_inference_speed(model, tokenizer, prompt, device, num_runs=5, max_new_tokens=50, do_sample=False, temperature=1.0, top_p=1.0):
    """Measure inference speed and throughput"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Warmup
    with torch.no_grad():
        model.generate(
            **inputs, 
            max_new_tokens=10, 
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tokenizer.eos_token_id
        )
    
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                pad_token_id=tokenizer.eos_token_id
            )
        end = time.perf_counter()
        times.append(end - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    tokens_per_sec = max_new_tokens / avg_time
    
    # Get generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return {
        "avg_time_sec": float(avg_time),
        "std_time_sec": float(std_time),
        "tokens_per_sec": float(tokens_per_sec),
        "generated_text": generated_text
    }

def measure_logits_quality(model_orig, model_comp, tokenizer, prompts, device):
    """Measure quality by comparing logits"""
    results = []
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            out_orig = model_orig(**inputs)
            out_comp = model_comp(**inputs)
        
        logits_orig = out_orig.logits
        logits_comp = out_comp.logits
        
        # Calculate metrics
        mse = torch.nn.functional.mse_loss(logits_orig, logits_comp).item()
        max_diff = torch.max(torch.abs(logits_orig - logits_comp)).item()
        
        # Cosine similarity
        logits_orig_flat = logits_orig.flatten()
        logits_comp_flat = logits_comp.flatten()
        cos_sim = torch.nn.functional.cosine_similarity(
            logits_orig_flat.unsqueeze(0), 
            logits_comp_flat.unsqueeze(0)
        ).item()
        
        results.append({
            "prompt": prompt,
            "logits_mse": float(mse),
            "max_logit_diff": float(max_diff),
            "cosine_similarity": float(cos_sim)
        })
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Benchmark GPT-Neo 1.3B original vs compressed")
    parser.add_argument("--original_dir", type=str, required=True)
    parser.add_argument("--compressed_dir", type=str, required=True)
    parser.add_argument("--output_json", type=str, default="benchmark_results_gpt_neo.json")
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()
    
    device = torch.device("cpu")  # Force CPU for fair comparison
    
    print("=" * 80)
    print("GPT-Neo 1.3B Benchmark: Original vs Compressed")
    print("=" * 80)
    
    # Test prompts
    prompts = [
        "The meaning of life is",
        "Artificial intelligence will",
        "In the future, technology",
    ]
    
    results = {
        "model": "EleutherAI/gpt-neo-1.3B",
        "compression_config": {
            "rank": 128,
            "quant_bits": 8,
            "min_dim": 1024
        },
        "device": str(device),
        "num_runs": args.num_runs,
        "max_new_tokens": args.max_new_tokens,
        "sampling": {
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p
        }
    }
    
    # ========== ORIGINAL MODEL ==========
    print("\n[1/3] Loading ORIGINAL model...")
    mem_before = get_memory_usage_mb()
    
    tokenizer = AutoTokenizer.from_pretrained(args.original_dir)
    model_orig = AutoModelForCausalLM.from_pretrained(args.original_dir).to(device)
    model_orig.eval()
    
    mem_after = get_memory_usage_mb()
    mem_orig = mem_after - mem_before
    
    print(f"   Memory used: {mem_orig:.2f} MB")
    
    # Measure speed
    print("   Measuring inference speed...")
    speed_orig = measure_inference_speed(
        model_orig, tokenizer, prompts[0], device, 
        args.num_runs, args.max_new_tokens,
        args.do_sample, args.temperature, args.top_p
    )
    
    results["original"] = {
        "memory_mb": float(mem_orig),
        "inference": speed_orig
    }
    
    print(f"   Speed: {speed_orig['tokens_per_sec']:.2f} tokens/sec")
    
    # ========== COMPRESSED MODEL ==========
    print("\n[2/3] Loading COMPRESSED model...")
    
    # Clear memory
    del model_orig
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    import gc
    gc.collect()
    
    import time
    time.sleep(1)  # Allow OS to update memory stats
    
    mem_before = get_memory_usage_mb()
    
    model_comp = AutoModelForCausalLM.from_pretrained(args.original_dir).to(device)
    model_comp.eval()
    
    num_patched, num_linear = patch_model_with_compressed_linears(
        model_comp, 
        Path(args.compressed_dir),
        quant_bits=8,
        verbose=False
    )
    
    gc.collect()  # Clean up after patching
    
    mem_after = get_memory_usage_mb()
    mem_comp = mem_after - mem_before
    
    print(f"   Patched: {num_patched}/{num_linear} layers")
    print(f"   Memory used: {mem_comp:.2f} MB")
    
    # Measure speed
    print("   Measuring inference speed...")
    speed_comp = measure_inference_speed(
        model_comp, tokenizer, prompts[0], device,
        args.num_runs, args.max_new_tokens,
        args.do_sample, args.temperature, args.top_p
    )
    
    results["compressed"] = {
        "memory_mb": float(mem_comp),
        "num_patched_layers": num_patched,
        "num_total_layers": num_linear,
        "inference": speed_comp
    }
    
    print(f"   Speed: {speed_comp['tokens_per_sec']:.2f} tokens/sec")
    
    # ========== QUALITY COMPARISON ==========
    print("\n[3/3] Measuring quality (logits comparison)...")
    
    # Reload original for comparison
    model_orig = AutoModelForCausalLM.from_pretrained(args.original_dir).to(device)
    model_orig.eval()
    
    quality_results = measure_logits_quality(
        model_orig, model_comp, tokenizer, prompts, device
    )
    
    results["quality"] = quality_results
    
    # Calculate averages
    avg_mse = np.mean([r["logits_mse"] for r in quality_results])
    avg_cos_sim = np.mean([r["cosine_similarity"] for r in quality_results])
    
    print(f"   Avg Logits MSE: {avg_mse:.6e}")
    print(f"   Avg Cosine Similarity: {avg_cos_sim:.6f}")
    
    # ========== SUMMARY ==========
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    memory_reduction = ((mem_orig - mem_comp) / mem_orig) * 100
    speed_change = ((speed_comp['tokens_per_sec'] - speed_orig['tokens_per_sec']) / speed_orig['tokens_per_sec']) * 100
    
    print(f"Memory Reduction: {memory_reduction:.1f}%")
    print(f"Speed Change: {speed_change:+.1f}%")
    print(f"Quality (Cosine Sim): {avg_cos_sim:.4f}")
    
    results["summary"] = {
        "memory_reduction_percent": float(memory_reduction),
        "speed_change_percent": float(speed_change),
        "avg_logits_mse": float(avg_mse),
        "avg_cosine_similarity": float(avg_cos_sim)
    }
    
    # Save results
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {args.output_json}")
    print("=" * 80)

if __name__ == "__main__":
    main()
