from pathlib import Path
import os

p = Path("models/original/gpt-neo-1.3B/model.safetensors.index.json")
print(f"CWD: {os.getcwd()}")
print(f"Path: {p}")
print(f"Absolute: {p.absolute()}")
print(f"Exists: {p.exists()}")
