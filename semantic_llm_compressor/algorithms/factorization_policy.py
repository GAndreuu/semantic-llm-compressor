# semantic_llm_compressor/algorithms/factorization_policy.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
from ..config import CompressionConfig


@dataclass
class FactorizationPolicy:
    """
    Ultra-conservative compression policy.
    
    After extensive testing, we found that most attention layers are highly
    sensitive to SVD compression at rank 256-384. This policy takes a minimal
    approach: compress ONLY k_proj layers, skip everything else.
    
    Rationale:
    - k_proj has shown the most stability under compression
    - q_proj, v_proj: High sensitivity, contributes to output degradation
    - out_proj: Extremely sensitive (negative cosines, norm explosions)
    - MLP: 78% SVD error at rank 128, 46% at rank 512
    - Embeddings/lm_head: Critical for quality
    
    This creates a stable baseline with ~24 compressed layers (one k_proj per block).
    """

    config: CompressionConfig

    def _is_large_enough(self, tensor: torch.Tensor) -> bool:
        """Check if tensor is large enough to benefit from compression."""
        if tensor is None or tensor.dim() != 2:
            return False
        
        out_features, in_features = tensor.shape
        min_dim = self.config.min_dim_for_compression
        
        return (out_features >= min_dim and in_features >= min_dim)

    def should_compress(self, weight_name: str, tensor: torch.Tensor) -> bool:
        """
        Ultra-conservative compression decision.
        
        ONLY compresses: attn.attention.k_proj (if large enough)
        SKIPS everything else: q_proj, v_proj, out_proj, MLP, embeddings, etc.
        
        This minimal approach ensures stability while still achieving some compression.
        """
        
        # Safety checks
        if tensor is None:
            return False
        
        if not self._is_large_enough(tensor):
            return False
        
        # 1) NEVER compress these critical components
        never_compress = [
            "wte",                 # Token embeddings - critical for vocabulary
            "wpe",                 # Position embeddings - critical for position info
            "ln_f",                # Final layer norm - critical for output distribution
            "lm_head",             # Output head - critical for logits
            "mlp.c_fc",            # MLP first layer - 78% SVD error at rank 128
            "mlp.c_proj",          # MLP second layer - equally sensitive
            "h.4.attn.attention.k_proj", # Toxic layer causing repetition (found via sensitivity analysis)
        ]
        
        for pattern in never_compress:
            if pattern in weight_name:
                return False
        
        # 2) SKIP highly sensitive attention projections
        #    Based on activation analysis showing negative cosines and norm explosions
        skip_attention = [
            "attn.attention.out_proj",  # Multiple instances with cosine < 0
            "attn.attention.q_proj",    # High sensitivity, low stability
            "attn.attention.v_proj",    # High sensitivity, low stability
        ]
        
        for pattern in skip_attention:
            if pattern in weight_name:
                return False
        
        # 3) ONLY compress k_proj (most stable under compression)
        if "attn.attention.k_proj" in weight_name:
            return True
        
        # 4) Everything else stays dense
        return False

    def choose_rank(self, weight_name: str, tensor: torch.Tensor) -> int:
        """
        Choose rank for compression.
        
        Since we only compress k_proj, we can use adaptive ranks:
        - First/last blocks: Higher rank (384) for stability
        - Middle blocks: Base rank (256) or higher
        """
        out_features, in_features = tensor.shape
        base_rank = self.config.rank
        
        # For k_proj in first or last blocks, use higher rank
        if "attn.attention.k_proj" in weight_name:
            # First 2 and last 2 blocks get extra high rank
            if any(f"h.{i}." in weight_name for i in [0, 1, 22, 23]):
                return 384  # Extra stability for boundary blocks
            
            # Large matrices get higher rank
            if out_features >= 2048 and in_features >= 2048:
                return 256  # Standard high rank for 2048×2048
            
            return 192  # Smaller k_proj layers
        
        # This shouldn't be reached since should_compress only allows k_proj
        return base_rank
