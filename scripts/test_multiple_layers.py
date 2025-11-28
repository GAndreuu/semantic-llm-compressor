# scripts/test_multiple_layers.py

"""
Testa múltiplas camadas de uma vez e salva resultados em JSON.
"""

import json
from pathlib import Path
from test_compressed_vs_dense import test_single_layer


def main():
    model_dir = "models/original/gpt-neo-1.3B"
    compressed_dir = "models/compressed/gpt-neo-1.3B-r256-adaptive"
    
    # Lista de camadas para testar
    layers_to_test = [
        # Primeira camada - Attention
        "transformer.h.0.attn.attention.out_proj.weight",
        "transformer.h.0.attn.attention.k_proj.weight",
        "transformer.h.0.attn.attention.q_proj.weight",
        "transformer.h.0.attn.attention.v_proj.weight",
        
        # Camada do meio - Attention
        "transformer.h.11.attn.attention.out_proj.weight",
        "transformer.h.11.attn.attention.k_proj.weight",
        
        # Última camada - Attention
        "transformer.h.23.attn.attention.out_proj.weight",
        "transformer.h.23.attn.attention.k_proj.weight",
    ]
    
    results = []
    
    for weight_name in layers_to_test:
        print(f"\n{'='*80}")
        print(f"Testing: {weight_name}")
        print('='*80)
        
        try:
            result = test_single_layer(
                model_dir=model_dir,
                compressed_dir=compressed_dir,
                weight_name=weight_name,
                batch_size=8,
            )
            results.append(result)
        except Exception as e:
            print(f"❌ ERROR: {e}")
            results.append({
                "layer": weight_name,
                "error": str(e),
            })
    
    # Save results
    output_file = "layer_quality_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"SUMMARY - {len(results)} layers tested")
    print('='*80)
    
    for r in results:
        if "error" in r:
            print(f"❌ {r['layer']}: ERROR")
        else:
            cos = r['cosine']
            symbol = "✅" if cos > 0.95 else "⚠️" if cos > 0.90 else "❌"
            print(f"{symbol} {r['layer']}: cosine={cos:.4f}, rel_err={r['rel_error']:.2%}")
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
