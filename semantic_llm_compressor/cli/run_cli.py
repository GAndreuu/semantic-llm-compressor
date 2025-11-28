# semantic_llm_compressor/cli/run_cli.py

from __future__ import annotations
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from semantic_llm_compressor.runtime.patcher import (
    patch_model_with_compressed_linears,
)
from semantic_llm_compressor.logging_utils import setup_logging


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run inference using a compressed LLM (SVD + INT8)."
    )
    p.add_argument(
        "--compressed_dir",
        type=str,
        required=True,
        help="Directory with compressed safetensors shards.",
    )
    p.add_argument(
        "--original_dir",
        type=str,
        required=True,
        help="Directory with original HF model (for config, tokenizer, and non-compressed weights).",
    )
    p.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt for generation.",
    )
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="Number of tokens to generate.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run inference on.",
    )
    p.add_argument(
        "--quant_bits",
        type=int,
        default=8,
        help="Quantization bits used in compression (must match compressor).",
    )
    return p


def main():
    parser = build_argparser()
    args = parser.parse_args()

    setup_logging()

    compressed_dir = Path(args.compressed_dir)
    original_dir = Path(args.original_dir)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[RUN] Loading original model from: {original_dir}")
    model = AutoModelForCausalLM.from_pretrained(original_dir)
    tokenizer = AutoTokenizer.from_pretrained(original_dir)

    model.to(device)
    model.eval()

    print(f"[RUN] Patching model using compressed_dir={compressed_dir}")
    num_patched, num_linear = patch_model_with_compressed_linears(
        model=model,
        compressed_dir=compressed_dir,
        quant_bits=args.quant_bits,
        verbose=True,
    )

    print(f"[RUN] Patched {num_patched}/{num_linear} linear layers.")

    # Preparar input
    prompt = args.prompt
    print(f"[RUN] Prompt: {prompt!r}")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Geração determinística simples (sem sampling) para comparação mais fácil
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\n=== Generated Text ===")
    print(text)
    print("======================\n")


if __name__ == "__main__":
    main()
