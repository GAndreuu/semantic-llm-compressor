#!/usr/bin/env python
"""
debug_decoding_step.py

Compara, passo a passo, o comportamento de um modelo original vs. um modelo
patchado com camadas comprimidas (SVD + INT8).

Uso típico:

  python debug_decoding_step.py \
    --model_path EleutherAI/gpt-neo-1.3B \
    --compressed_dir models/compressed/gpt-neo-1.3B-r256-adaptive \
    --prompt "The meaning of life is"

Requer:
  - transformers
  - torch
  - seu pacote semantic_llm_compressor instalado em modo editable (-e)
"""

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# 🔧 AJUSTE AQUI se o patcher estiver em outro módulo.
try:
    from semantic_llm_compressor.runtime.patcher import (
        patch_model_with_compressed_linears,
    )
except ImportError:
    patch_model_with_compressed_linears = None
    print(
        "[WARN] Não consegui importar 'patch_model_with_compressed_linears'. "
        "Ajuste o import no topo deste arquivo para apontar para o patcher real."
    )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


@dataclass
class TensorStats:
    name: str
    shape: Tuple[int, ...]
    mse: float
    max_diff: float
    cosine: float
    l2_original: float
    l2_compressed: float


def flatten_tensor(x: torch.Tensor) -> torch.Tensor:
    """Achata o tensor em 1D (para cosine, etc.)."""
    return x.reshape(-1)


def compute_pair_stats(
    name: str, t_orig: torch.Tensor, t_comp: torch.Tensor
) -> TensorStats:
    """Computa métricas entre duas ativações."""
    with torch.no_grad():
        a = flatten_tensor(t_orig).float()
        b = flatten_tensor(t_comp).float()

        diff = a - b
        mse = diff.pow(2).mean().item()
        max_diff = diff.abs().max().item()

        # Cosine similarity – cuida do caso de norma zero
        norm_a = a.norm(p=2)
        norm_b = b.norm(p=2)
        if norm_a.item() == 0 or norm_b.item() == 0:
            cosine = float("nan")
        else:
            cosine = F.cosine_similarity(a, b, dim=0).item()

        return TensorStats(
            name=name,
            shape=tuple(t_orig.shape),
            mse=mse,
            max_diff=max_diff,
            cosine=cosine,
            l2_original=norm_a.item(),
            l2_compressed=norm_b.item(),
        )


def pretty_print_stats(stats: List[TensorStats], title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"{'Layer':50s} {'Shape':20s} {'MSE':>10s} {'MaxDiff':>10s} {'Cosine':>10s}")
    print("-" * 80)
    for s in stats:
        shape_str = "x".join(str(d) for d in s.shape)
        print(
            f"{s.name:50s} {shape_str:20s} "
            f"{s.mse:10.4f} {s.max_diff:10.4f} {s.cosine:10.4f}"
        )


# ---------------------------------------------------------------------
# Hook registration
# ---------------------------------------------------------------------


def register_hooks_for_patterns(
    model: torch.nn.Module, patterns: List[str], store: Dict[str, torch.Tensor]
) -> List[torch.utils.hooks.RemovableHandle]:
    """
    Registra forward hooks em todos os submódulos cujo nome contenha
    algum dos padrões especificados.
    """
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def make_hook(name):
        def hook(_module, _inp, output):
            # Algumas camadas retornam tupla (output, extra); pegamos só o primeiro
            if isinstance(output, (list, tuple)):
                out = output[0]
            else:
                out = output
            store[name] = out.detach().cpu()

        return hook

    for name, module in model.named_modules():
        if any(pat in name for pat in patterns):
            h = module.register_forward_hook(make_hook(name))
            handles.append(h)
    return handles


# ---------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------


def run_debug_step(
    model_path: str,
    compressed_dir: str,
    prompt: str,
    device: str = None,
    max_new_tokens: int = 1,
    save_json: str = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Device: {device}")

    # 1. Tokenizer
    print("[INFO] Carregando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Modelos
    print("[INFO] Carregando modelo ORIGINAL...")
    orig_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map=None,
    ).to(device)
    orig_model.eval()

    print("[INFO] Carregando modelo COMPRIMIDO (base igual + patch)...")
    comp_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map=None,
    ).to(device)
    comp_model.eval()

    if patch_model_with_compressed_linears is None:
        raise RuntimeError(
            "patch_model_with_compressed_linears não importado. "
            "Ajuste o import no topo deste arquivo."
        )

    print(f"[INFO] Aplicando patch do compressor: {compressed_dir}")
    patch_model_with_compressed_linears(
        comp_model,
        Path(compressed_dir),
        quant_bits=8,
        verbose=False,
    )

    # 3. Hooks
    # Alvos típicos do GPT-Neo (com base nos nomes que você já mediu):
    #   transformer.h.N.attn.attention.{q_proj,k_proj,v_proj,out_proj}
    #   lm_head
    target_patterns = [
        "attn.attention.q_proj",
        "attn.attention.k_proj",
        "attn.attention.v_proj",
        "attn.attention.out_proj",
        "lm_head",
    ]

    orig_activations: Dict[str, torch.Tensor] = {}
    comp_activations: Dict[str, torch.Tensor] = {}

    print("[INFO] Registrando hooks em camadas de atenção + lm_head...")
    orig_handles = register_hooks_for_patterns(orig_model, target_patterns, orig_activations)
    comp_handles = register_hooks_for_patterns(comp_model, target_patterns, comp_activations)

    # 4. Preparar input
    print(f"[INFO] Tokenizando prompt: {prompt!r}")
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        truncation=True,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # 5. Forward único (uma etapa de "decoding" sem generate)
    with torch.no_grad():
        print("[INFO] Forward no modelo original...")
        out_orig = orig_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

        print("[INFO] Forward no modelo comprimido...")
        out_comp = comp_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

    # Remove hooks para liberar memória
    for h in orig_handles + comp_handles:
        h.remove()

    # 6. Comparar logits finais (último token)
    logits_orig = out_orig.logits[:, -1, :].detach().cpu()
    logits_comp = out_comp.logits[:, -1, :].detach().cpu()

    logits_stats = compute_pair_stats("logits_last_token", logits_orig, logits_comp)

    # 7. Comparar ativações das camadas
    layer_stats: List[TensorStats] = []
    all_keys = sorted(set(orig_activations.keys()) | set(comp_activations.keys()))

    for name in all_keys:
        if name not in orig_activations or name not in comp_activations:
            print(f"[WARN] Ativação ausente em um dos modelos: {name}")
            continue
        s = compute_pair_stats(name, orig_activations[name], comp_activations[name])
        layer_stats.append(s)

    # 8. Impressão
    pretty_print_stats(layer_stats, "Per-layer activation comparison (original vs compressed)")
    pretty_print_stats([logits_stats], "Logits (last token) comparison")

    print("\nResumo rápido:")
    print(f"  • Cosine médio camadas: "
          f"{sum(s.cosine for s in layer_stats) / max(len(layer_stats),1):.3f}")
    print(f"  • Cosine logits último token: {logits_stats.cosine:.3f}")
    print(f"  • MSE médio camadas: "
          f"{sum(s.mse for s in layer_stats) / max(len(layer_stats),1):.3f}")
    print(f"  • Prompt testado: {prompt!r}")

    # 9. Opcional: salvar em JSON pra análises futuras
    if save_json is not None:
        payload = {
            "model": model_path,
            "compressed_dir": compressed_dir,
            "prompt": prompt,
            "device": device,
            "logits": {
                "mse": logits_stats.mse,
                "max_diff": logits_stats.max_diff,
                "cosine": logits_stats.cosine,
                "shape": logits_stats.shape,
            },
            "layers": [
                {
                    "name": s.name,
                    "shape": s.shape,
                    "mse": s.mse,
                    "max_diff": s.max_diff,
                    "cosine": s.cosine,
                    "l2_original": s.l2_original,
                    "l2_compressed": s.l2_compressed,
                }
                for s in layer_stats
            ],
        }
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[INFO] Resultado salvo em: {save_json}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Debug de um passo de decodificação: original vs comprimido."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path ou ID HF do modelo original (ex: EleutherAI/gpt-neo-1.3B).",
    )
    parser.add_argument(
        "--compressed_dir",
        type=str,
        required=True,
        help="Diretório com os shards/fatores comprimidos.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt usado para o forward de comparação.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cpu ou cuda. Se omitido, escolhe automaticamente.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1,
        help="(Reservado) Número de tokens novos — aqui só usamos 1 passo de forward.",
    )
    parser.add_argument(
        "--save_json",
        type=str,
        default=None,
        help="Se fornecido, salva resultados em JSON neste caminho.",
    )

    args = parser.parse_args()

    run_debug_step(
        model_path=args.model_path,
        compressed_dir=args.compressed_dir,
        prompt=args.prompt,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        save_json=args.save_json,
    )


if __name__ == "__main__":
    main()
