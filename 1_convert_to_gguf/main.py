import os
import os.path
import subprocess
import sys
from pathlib import Path

BASE_MODEL = os.getenv('BASE_MODEL', 'Qwen2.5-1.5B-instruct')
OUT_FORMAT = os.getenv('OUT_FORMAT', 'f16')
LORA_PATH = f"outputs/{BASE_MODEL}/final_adapter/"
OUT_BASE_MODEL_FILE = Path(f"exported_models/{BASE_MODEL}_{OUT_FORMAT}.gguf")
OUT_LORA_ADAPTER_FILE = Path(f"exported_models/{BASE_MODEL}_LORA_{OUT_FORMAT}.gguf")
LLAMA_CPP_DIR = Path(os.getenv('LLAMA_CPP_DIR', 'llama.cpp/'))
LLAMA_BIN_DIR = Path(os.getenv('LLAMA_BIN_DIR', 'llama.cpp/bin'))
sys.path.append(str(LLAMA_CPP_DIR))

converter_path = str(LLAMA_CPP_DIR / "convert_hf_to_gguf.py")

if os.path.isfile(converter_path):
    model_f = f'models/{BASE_MODEL}'
    if not os.path.isfile(OUT_BASE_MODEL_FILE):
        print(f"===> Converting models/{BASE_MODEL} to the .gguf format")
        subprocess.run([
            sys.executable, str(LLAMA_CPP_DIR / "convert_hf_to_gguf.py"),
            model_f,
            '--outfile', OUT_BASE_MODEL_FILE,
            '--outtype', OUT_FORMAT
        ], check=True)

    print(f"===> Converting a LoRA adapter to the .gguf format. {LORA_PATH}")
    subprocess.run([
        sys.executable, str(LLAMA_CPP_DIR / "convert_lora_to_gguf.py"),
        LORA_PATH,
        '--outfile', OUT_LORA_ADAPTER_FILE,
        '--outtype', OUT_FORMAT,
        '--base', f'models/{BASE_MODEL}'
    ], check=True)

    print(f"Base model: {OUT_BASE_MODEL_FILE}")
    print(f"LoRA adapter:  {OUT_LORA_ADAPTER_FILE}")
else:
    print(f"===> Error: can't find converter: {converter_path}")

print(f"===> Quantization q4_k_m for. {OUT_BASE_MODEL_FILE}")

quatizator_path = str(LLAMA_BIN_DIR / "llama-quantize.exe")

BASE_MODEL_Q4 = f'exported_models/{BASE_MODEL}_q4_k_m.gguf'
if not os.path.isfile(BASE_MODEL_Q4):
    if os.path.isfile(quatizator_path):
        subprocess.run([
            quatizator_path,
            OUT_BASE_MODEL_FILE,
            f"exported_models/{BASE_MODEL}_q4_k_m.gguf",
            'q4_k_m'
        ], check=True)
    else:
        print(f"===> Error: can't find quantizator: {quatizator_path}")

print('\n Ready!')