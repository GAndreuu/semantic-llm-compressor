import argparse
import torch
import json
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from semantic_llm_compressor.runtime.patcher import patch_model_with_compressed_linears
from semantic_llm_compressor.runtime.compressed_layers import CompressedLinear

def main():
    parser = argparse.ArgumentParser(description="Identify toxic compressed layers")
    parser.add_argument("--original_dir", type=str, required=True)
    parser.add_argument("--compressed_dir", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="The meaning of life is")
    parser.add_argument("--output_json", type=str, default="layer_sensitivity_results.json")
    args = parser.parse_args()

    device = torch.device("cpu")
    
    print(f"Loading model from {args.original_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.original_dir)
    model = AutoModelForCausalLM.from_pretrained(args.original_dir).to(device)
    model.eval()

    # Keep a copy of original weights for the layers we are interested in
    # Actually, it's easier to just load the compressed model and revert specific layers if we have the original model handy?
    # No, memory is tight.
    # Strategy:
    # 1. Load model.
    # 2. Patch ALL layers.
    # 3. For each patched layer:
    #    a. Revert to original (we need to store original weights before patching? or reload?)
    #    b. Measure quality.
    #    c. Re-patch.
    
    # Better strategy for memory:
    # 1. Load model.
    # 2. Identify target layers (k_proj).
    # 3. Store original weights of ONLY these layers in RAM (it's only 24 layers, ~100MB).
    # 4. Patch model.
    # 5. Iterate.

    print("Identifying target layers...")
    target_layers = {}
    for name, module in model.named_modules():
        if "attn.attention.k_proj" in name:
            target_layers[name] = module.weight.detach().clone()
            print(f"  Stored original weights for {name}")

    print(f"Stored {len(target_layers)} layers.")

    print("Patching model...")
    num_patched, _ = patch_model_with_compressed_linears(
        model, 
        Path(args.compressed_dir),
        quant_bits=8,
        verbose=False
    )
    print(f"Patched {num_patched} layers.")

    # Baseline (All Compressed)
    print("\nMeasuring Baseline (All Compressed)...")
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out_base = model(**inputs)
        logits_base = out_base.logits
        
        # Generate text
        gen_base = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        text_base = tokenizer.decode(gen_base[0], skip_special_tokens=True)

    print(f"Baseline Text: {text_base}")

    results = []
    
    # Iterate
    print("\nTesting individual layer reversion...")
    for name, original_weight in target_layers.items():
        # Find the module
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        layer_name = parts[-1]
        
        # Get compressed module
        compressed_module = getattr(parent, layer_name)
        if not isinstance(compressed_module, CompressedLinear):
            print(f"Skipping {name} (not compressed)")
            continue

        # Revert to original (create new Linear)
        original_module = torch.nn.Linear(
            original_weight.shape[1], 
            original_weight.shape[0], 
            bias=compressed_module.bias is not None
        )
        original_module.weight.data = original_weight
        if compressed_module.bias is not None:
            original_module.bias.data = compressed_module.bias
        
        # Swap
        setattr(parent, layer_name, original_module)
        
        # Measure
        with torch.no_grad():
            out_curr = model(**inputs)
            logits_curr = out_curr.logits
            
            # MSE vs Baseline (Higher means this layer was changing things a lot)
            # But we want to know if it improves QUALITY.
            # We don't have ground truth logits here easily unless we ran original model first.
            # Let's just look at generated text changes? Or entropy?
            
            # Let's generate text
            gen_curr = model.generate(**inputs, max_new_tokens=20, do_sample=False)
            text_curr = tokenizer.decode(gen_curr[0], skip_special_tokens=True)
        
        # Restore compressed
        setattr(parent, layer_name, compressed_module)
        
        print(f"Reverted {name}: {text_curr}")
        
        results.append({
            "layer": name,
            "text": text_curr,
            "changed_from_baseline": text_curr != text_base
        })

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {args.output_json}")

if __name__ == "__main__":
    main()
