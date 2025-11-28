# semantic_llm_compressor/algorithms/svd_compressor.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class SVDDecomposer:
    """
    Responsável por aplicar SVD truncado em tensores 2D.

    Dado um tensor W [out_dim, in_dim], produz:
    - U [out_dim, r]
    - S [r]
    - Vh [r, in_dim]
    onde r é o rank alvo (truncado se maior que min(out_dim, in_dim)).
    """

    rank: int

    def decompose(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Aplica SVD truncado ao tensor de pesos.

        Parâmetros
        ----------
        weight : torch.Tensor
            Tensor 2D de pesos, formato [out_dim, in_dim].

        Retorna
        -------
        U : torch.Tensor
            Matriz U truncada, shape [out_dim, r].
        S : torch.Tensor
            Vetor de singular values truncado, shape [r].
        Vh : torch.Tensor
            Matriz V^T truncada, shape [r, in_dim].
        """
        if weight.dim() != 2:
            raise ValueError(f"SVDDecomposer espera tensor 2D, recebeu shape {tuple(weight.shape)}")

        # Para estabilidade, trabalhamos em float32 durante a decomposição
        # (mesmo se o tensor original for float16)
        dtype_original = weight.dtype
        device = weight.device
        w_f32 = weight.to(torch.float32)

        # SVD "econômico" (full_matrices=False) para evitar matrizes gigantes
        # U: [m, k], S: [k], Vh: [k, n] onde k = min(m, n)
        U, S, Vh = torch.linalg.svd(w_f32, full_matrices=False)

        k = S.shape[0]
        r = min(self.rank, k)

        # Truncamento
        U_r = U[:, :r].contiguous()
        S_r = S[:r].contiguous()
        Vh_r = Vh[:r, :].contiguous()

        # === NORM CORRECTION ===
        # Prevents energy explosion/collapse in compressed weights
        # Matches Frobenius norm of approximation to original
        
        # Compute ||W_orig||_F
        norm_original = torch.norm(w_f32, p='fro')
        
        # Compute ||W_hat||_F where W_hat = U_r @ diag(S_r) @ Vh_r
        # Using property: ||U @ diag(S) @ Vh||_F = ||S||_2 for orthonormal U, Vh
        norm_approx = torch.norm(S_r, p=2)
        
        # Scale singular values to match original norm
        if norm_approx > 1e-8:  # Avoid division by zero
            scale_factor = norm_original / norm_approx
            S_r = S_r * scale_factor
        
        # This prevents issues like:
        # - l2_original: 80, l2_compressed: 2079 (26× explosion!)
        # - Negative cosine similarities
        # - Activation instability

        # Opcional: devolver no dtype original (por ex. float16) para alinhar com o restante do pipeline
        U_r = U_r.to(dtype_original).to(device)
        S_r = S_r.to(dtype_original).to(device)
        Vh_r = Vh_r.to(dtype_original).to(device)

        return U_r, S_r, Vh_r
