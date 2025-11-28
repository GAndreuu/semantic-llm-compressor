import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from semantic_llm_compressor.runtime.patcher import patch_model_with_compressed_linears

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_dir", type=str, required=True)
    parser.add_argument("--compressed_dir", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.original_dir)
    
    prompts = [
        "The meaning of life is",
        "Artificial intelligence will",
        "In the future, technology"
    ]

    with open("generation_comparison.txt", "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("COMPARATIVE GENERATION ANALYSIS\n")
        f.write("="*80 + "\n")

        # 1. Original Model
        print("\nLoading ORIGINAL model...")
        model = AutoModelForCausalLM.from_pretrained(args.original_dir).to(device)
        model.eval()

        original_outputs = {}
        
        f.write("\n--- Original Generation ---\n")
        for prompt in prompts:
            # Greedy
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            out_greedy = model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            text_greedy = tokenizer.decode(out_greedy[0], skip_special_tokens=True)
            
            # Sampling
            torch.manual_seed(42) # Fixed seed for reproducibility
            out_sample = model.generate(**inputs, max_new_tokens=20, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
            text_sample = tokenizer.decode(out_sample[0], skip_special_tokens=True)
            
            original_outputs[prompt] = {"greedy": text_greedy, "sample": text_sample}
            f.write(f"\nPrompt: {prompt}\n")
            f.write(f"Greedy: {text_greedy}\n")
            f.write(f"Sample: {text_sample}\n")

        # 2. Compressed Model
        print("\nLoading COMPRESSED model...")
        # Reload to be clean
        del model
        import gc
        gc.collect()
        
        model = AutoModelForCausalLM.from_pretrained(args.original_dir).to(device)
        model.eval()
        patch_model_with_compressed_linears(model, Path(args.compressed_dir), quant_bits=8, verbose=False)

        f.write("\n--- Compressed Generation ---\n")
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Greedy
            out_greedy = model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            text_greedy = tokenizer.decode(out_greedy[0], skip_special_tokens=True)
            
            # Sampling
            torch.manual_seed(42) # Same seed
            out_sample = model.generate(**inputs, max_new_tokens=20, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
            text_sample = tokenizer.decode(out_sample[0], skip_special_tokens=True)
            
            f.write(f"\nPrompt: {prompt}\n")
            f.write(f"Greedy: {text_greedy}\n")
            f.write(f"Sample: {text_sample}\n")
            
            # Compare
            orig_greedy = original_outputs[prompt]["greedy"]
            if text_greedy == orig_greedy:
                f.write(">> GREEDY MATCH: ✅ Perfect\n")
            else:
                f.write(">> GREEDY MATCH: ❌ Diverged\n")
                
            orig_sample = original_outputs[prompt]["sample"]
            if text_sample == orig_sample:
                f.write(">> SAMPLE MATCH: ✅ Perfect (Surprisingly)\n")
            else:
                f.write(">> SAMPLE MATCH: ❌ Diverged (Expected)\n")

if __name__ == "__main__":
    main()
