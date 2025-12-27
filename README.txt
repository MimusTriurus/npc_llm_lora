1. Установи зависимости:
   pip install -r requirements.txt
   (для torch используй колёсико с сайта PyTorch под CUDA 12.1)
   pip install torch==2.5.1+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

2. Запусти обучение:
   python train_qlora_qwen.py

3. После окончания:
   python merge_lora_qwen.py

4. Для экспорта в GGUF:
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp
   python convert-hf-to-gguf.py ../qwen_skyrim_merged --outdir ./gguf_out
   ./quantize ./gguf_out/qwen_skyrim_merged.gguf ./gguf_out/qwen_skyrim_merged.Q4_K_M.gguf Q4_K_M
