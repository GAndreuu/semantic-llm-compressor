# semantic_llm_compressor/eval/metrics.py

from __future__ import annotations

import torch
import torch.nn.functional as F


def weight_mse(original: torch.Tensor, approx: torch.Tensor) -> float:
    """
    MSE entre pesos originais e aproximados.
    """
    if original.shape != approx.shape:
        raise ValueError(f"Shapes incompatíveis: {original.shape} vs {approx.shape}")
    
    return F.mse_loss(original, approx).item()


def activation_mse(original_model: torch.nn.Module, compressed_model: torch.nn.Module, input_ids: torch.Tensor) -> float:
    """
    MSE entre ativações (logits finais) originais e aproximadas para o mesmo input.
    """
    original_model.eval()
    compressed_model.eval()
    
    with torch.no_grad():
        out_orig = original_model(input_ids).logits
        out_comp = compressed_model(input_ids).logits
        
    return F.mse_loss(out_orig, out_comp).item()
