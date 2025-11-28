# scripts/4_prepare_gpt_neo.py

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

def main():
    model_id = "EleutherAI/gpt-neo-1.3B"
    out_dir = "models/original/gpt-neo-1.3B"
    
    print(f"Baixando tokenizer de {model_id}...")
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.save_pretrained(out_dir)

    print(f"Baixando modelo de {model_id}...")
    # Usando float32 para garantir compatibilidade com CPU se necessário, 
    # mas float16 é preferível se tiver RAM suficiente.
    # O user sugeriu float16, mas com fallback se der erro.
    # Vamos tentar float32 para ser safe na CPU, já que 1.3B cabe em ~6GB RAM.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32, 
            low_cpu_mem_usage=True,
            device_map=None,
        )
        model.save_pretrained(out_dir, safe_serialization=True)
        print("✅ Modelo salvo em", out_dir)
    except Exception as e:
        print(f"Erro ao baixar modelo: {e}")

if __name__ == "__main__":
    main()
