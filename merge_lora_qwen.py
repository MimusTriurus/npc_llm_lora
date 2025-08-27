from peft import PeftModel
from transformers import AutoModelForCausalLM

BASE_MODEL = "Qwen/Qwen2.5-1.5B"
LORA_ADAPTER = "./qwen_lora_skyrim"
OUTPUT_DIR = "./qwen_skyrim_merged"

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="cpu")
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER)
model = model.merge_and_unload()

model.save_pretrained(OUTPUT_DIR)
print(f"✅ Merged model saved at {OUTPUT_DIR}")
