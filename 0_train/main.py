import torch
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

model_name = os.getenv('MODEL_NAME', "Qwen/Qwen2.5-1.5B")

print(f'== Model name: {model_name}\n')

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

model.enable_input_require_grads()
model.config.use_cache = False
model.print_trainable_parameters()

dataset = load_dataset("json", data_files={"0_train": "data/dataset.jsonl"})

def format_example(example):
    return {
        "text": f"Instruction: {example['instruction']}\nAnswer: {example['output']}"
    }

dataset = dataset.map(format_example)

sft_config = SFTConfig(
    output_dir="../outputs",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    num_train_epochs=3,
    logging_steps=5,
    save_strategy="epoch",
    save_total_limit=2,
    bf16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    max_seq_length=512,
    packing=True,
    dataset_text_field="text",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["0_train"],
    tokenizer=tokenizer,
    args=sft_config,
)

trainer.train()

save_dir = "../outputs/final_adapter"
model.save_pretrained(save_dir, safe_serialization=False)
tokenizer.save_pretrained(save_dir)
