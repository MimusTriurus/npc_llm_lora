1. Запуск обучения в Docker:
   CUDA 12.8
   docker compose up

2. Для экспорта в GGUF:
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp
   python convert-hf-to-gguf.py ../qwen_skyrim_merged --outdir ./gguf_out
