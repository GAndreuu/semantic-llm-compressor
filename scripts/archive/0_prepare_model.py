# scripts/0_prepare_model.py

from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "sshleifer/tiny-gpt2"
OUTPUT_DIR = Path("models/original/tiny-gpt2")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Baixando modelo {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Salvando em formato safetensors...")
    model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Concluído. Arquivos em:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
