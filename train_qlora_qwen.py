import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# --- 1. Модель ---
model_name = "Qwen/Qwen2.5-1.5B"

# Токенизатор
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# --- 2. Квант (bnb 4-bit) ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # лучше bf16, чем fp16, на Ada GPU
)

# Загружаем модель с квантованием
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
)

# --- 3. LoRA ---
lora_config = LoraConfig(
    r=8,                           # базовый rank (можно 16, если данных станет больше)
    lora_alpha=32,                 # поднял для большей "силы" адаптера
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # больше охват → выше качество
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# --- 4. Датасет ---
dataset = load_dataset("json", data_files={"train": "data/dataset.jsonl"})

def format_example(example):
    return {
        "text": f"<|user|>\n{example['instruction']}\n<|assistant|>\n{example['output']}"
    }   # формат ближе к chat-style

dataset = dataset.map(format_example)

# --- 5. Аргументы обучения ---
training_args = TrainingArguments(
    per_device_train_batch_size=8,              # можно поднять, GPU 16GB потянет
    gradient_accumulation_steps=2,              # эквивалентный батч = 16
    warmup_ratio=0.1,                           # warmup как доля шагов вместо фикс. числа
    num_train_epochs=3,                         # >1 эпохи, иначе не учится
    learning_rate=2e-5,                         # пониже, чтобы не "затирал" веса
    lr_scheduler_type="cosine",                 # плавное снижение lr
    logging_steps=5,
    save_strategy="epoch",                      # сохранять по эпохам
    save_total_limit=2,
    fp16=False,                                 # отключаем fp16
    bf16=True,                                  # лучше использовать BF16 (на 5070Ti есть)
    optim="paged_adamw_32bit",                  # оптимайзер для 4bit
    output_dir="outputs",
    report_to="none",
    gradient_checkpointing=True,                # экономит VRAM, можно выше batch_size
)

# --- 6. Trainer ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
    args=training_args,
    max_seq_length=512,    # явное ограничение, если тексты короткие
    dataset_text_field="text",
    packing=True,          # объединяет короткие примеры → выше эффективность
)

trainer.train()