# semantic_llm_compressor/eval/eval_pipeline.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from .metrics import weight_mse, activation_mse
from ..logging_utils import get_logger

logger = get_logger(__name__)

@dataclass
class EvaluationResult:
    """
    Guarda resultados de avaliação (perplexidade, MSE, etc.).
    """
    metrics: Dict[str, float]


@dataclass
class EvaluationPipeline:
    """
    Pipeline de avaliação para modelos comprimidos.
    """

    def evaluate(self, original_model: Any, compressed_model: Any, tokenizer: Any, test_text: str = "Hello world") -> EvaluationResult:
        """
        Executa avaliação comparando modelo original e modelo comprimido.
        """
        logger.info("Iniciando avaliação...")
        
        metrics = {}
        
        # 1. Activation MSE (Logits)
        inputs = tokenizer(test_text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        mse = activation_mse(original_model, compressed_model, input_ids)
        metrics["logits_mse"] = mse
        logger.info(f"Logits MSE: {mse}")
        
        # 2. Perplexidade (Simulada/Simplificada)
        # Calcular loss no texto de teste
        with torch.no_grad():
            outputs_orig = original_model(input_ids, labels=input_ids)
            loss_orig = outputs_orig.loss.item()
            ppl_orig = torch.exp(outputs_orig.loss).item()
            
            outputs_comp = compressed_model(input_ids, labels=input_ids)
            loss_comp = outputs_comp.loss.item()
            ppl_comp = torch.exp(outputs_comp.loss).item()
            
        metrics["ppl_original"] = ppl_orig
        metrics["ppl_compressed"] = ppl_comp
        metrics["ppl_diff"] = ppl_comp - ppl_orig
        
        logger.info(f"PPL Original: {ppl_orig:.4f}, PPL Compressed: {ppl_comp:.4f}")
        
        return EvaluationResult(metrics=metrics)
