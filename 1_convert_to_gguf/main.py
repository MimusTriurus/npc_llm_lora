import os
import subprocess
import sys
from pathlib import Path

BASE_MODEL = "models/Qwen2.5-1.5B-instruct"
OUT_FORMAT = "q8_0"
LORA_PATH = r"outputs/final_adapter/"
EXPORT_DIR = Path("exported_models")
LLAMA_CPP_DIR = Path("D:/Projects/C++/llama.cpp/")
sys.path.append(str(LLAMA_CPP_DIR))

EXPORT_DIR.mkdir(exist_ok=True)

print("===> Конвертируем Qwen в .gguf")
subprocess.run([
    sys.executable, str(LLAMA_CPP_DIR / "convert_hf_to_gguf.py"),
    BASE_MODEL,
    "--outfile", str(EXPORT_DIR / "qwen-1.5b.gguf"),
    "--outtype", OUT_FORMAT
], check=True)

print("===> Конвертируем LoRA адаптер в .gguf")
subprocess.run([
    sys.executable, str(LLAMA_CPP_DIR / "convert_lora_to_gguf.py"),
    LORA_PATH,
    "--outfile", str(EXPORT_DIR / "npc_adapter.gguf"),
    "--outtype", OUT_FORMAT
], check=True)

print("\n✅ Готово!")
print(f"Базовая модель: {EXPORT_DIR / 'qwen-1.5b.gguf'}")
print(f"LoRA адаптер:  {EXPORT_DIR / 'npc_adapter.gguf'}")

print("\nПример запуска в llama.cpp:")
print(f"./main -m {EXPORT_DIR / 'qwen-1.5b.gguf'} --lora {EXPORT_DIR / 'npc_adapter.gguf'} -p \"Hello NPC!\"")