# scripts/compare_original_vs_compressed.py

import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from semantic_llm_compressor.runtime.patcher import patch_model_with_compressed_linears

def main():
    parser = argparse.ArgumentParser(description="Compare original vs compressed model.")
    parser.add_argument("--original_dir", type=str, required=True)
    parser.add_argument("--compressed_dir", type=str, required=True)
    parser.add_argument("--output_json", type=str, default="comparison_results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading original model from {args.original_dir}...")
    model_orig = AutoModelForCausalLM.from_pretrained(args.original_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.original_dir)
    model_orig.eval()

    print(f"Loading compressed model from {args.original_dir} + patch...")
    # Load another instance to patch
    model_comp = AutoModelForCausalLM.from_pretrained(args.original_dir).to(device)
    model_comp.eval()
    
    patch_model_with_compressed_linears(model_comp, Path(args.compressed_dir), verbose=True)

    prompts = [
        "The meaning of life is",
        "Once upon a time",
        "Python is a programming language that",
    ]

    results = []

    for prompt in prompts:
        print(f"\nProcessing prompt: {prompt!r}")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            out_orig = model_orig(**inputs)
            out_comp = model_comp(**inputs)
            
            # Generate text
            gen_orig = model_orig.generate(**inputs, max_new_tokens=20, do_sample=False)
            gen_comp = model_comp.generate(**inputs, max_new_tokens=20, do_sample=False)

        logits_orig = out_orig.logits[:, -1, :]
        logits_comp = out_comp.logits[:, -1, :]

        mse = torch.nn.functional.mse_loss(logits_orig, logits_comp).item()
        max_diff = torch.max(torch.abs(logits_orig - logits_comp)).item()
        
        text_orig = tokenizer.decode(gen_orig[0], skip_special_tokens=True)
        text_comp = tokenizer.decode(gen_comp[0], skip_special_tokens=True)

        print(f"  MSE: {mse:.6e}")
        print(f"  Max Diff: {max_diff:.6e}")
        print(f"  Orig: {text_orig!r}")
        print(f"  Comp: {text_comp!r}")

        results.append({
            "prompt": prompt,
            "logits_mse": mse,
            "max_logit_diff": max_diff,
            "output_original": text_orig,
            "output_compressed": text_comp,
        })

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output_json}")

if __name__ == "__main__":
    main()
