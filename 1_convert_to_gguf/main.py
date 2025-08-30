import os
import subprocess
import sys
from pathlib import Path

BASE_MODEL = os.getenv('BASE_MODEL', "models/Qwen2.5-1.5B-instruct")
OUT_FORMAT = os.getenv('OUT_FORMAT', "q8_0")
LORA_PATH = os.getenv('LORA_PATH', "outputs/final_adapter/")
EXPORT_DIR = Path(os.getenv('EXPORT_DIR', "exported_models"))
LLAMA_CPP_DIR = Path(os.getenv('LLAMA_CPP_DIR', "llama.cpp/"))
sys.path.append(str(LLAMA_CPP_DIR))

EXPORT_DIR.mkdir(exist_ok=True)

print(f"===> Converting {BASE_MODEL} to the .gguf format")
subprocess.run([
    sys.executable, str(LLAMA_CPP_DIR / "convert_hf_to_gguf.py"),
    BASE_MODEL,
    "--outfile", str(EXPORT_DIR / "base_model.gguf"),
    "--outtype", OUT_FORMAT
], check=True)

print(f"===> Converting a LoRA adapter to the .gguf format")
subprocess.run([
    sys.executable, str(LLAMA_CPP_DIR / "convert_lora_to_gguf.py"),
    LORA_PATH,
    "--outfile", str(EXPORT_DIR / "lora_adapter.gguf"),
    "--outtype", OUT_FORMAT
], check=True)

print("\n Ready!")
print(f"Base model: {EXPORT_DIR / 'base_model.gguf'}")
print(f"LoRA adapter:  {EXPORT_DIR / 'lora_adapter.gguf'}")

with open(EXPORT_DIR / "lora_adapter.gguf", 'w') as f:
    f.writelines(
        [
            BASE_MODEL,
            OUT_FORMAT,
        ]
    )