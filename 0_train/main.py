import json
from collections import defaultdict
import random

import torch
import os
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

def normalize_dataset(data_files: str, dataset_group_size: int = -1) -> Dataset:
    print(data_files)
    dataset: Dataset = load_dataset(
        "json",
        data_files=data_files,
        split="train",
    )

    def extract_action_name(example):
        data = json.loads(example["messages"][-1]["content"])
        return {"action_name": data["action"]["name"]}

    dataset = dataset.map(extract_action_name)

    indices_by_action = defaultdict(list)

    for idx, example in enumerate(dataset):
        indices_by_action[example["action_name"]].append(idx)

    sizes = {k: len(v) for k, v in indices_by_action.items()}
    print("Sizes data by Action class:", sizes)

    target_size = dataset_group_size if dataset_group_size != -1 else int(sum(sizes.values()) / len(sizes))

    print("Target size for each Action class:", target_size)

    balanced_groups = {}

    for action_name, idxs in indices_by_action.items():
        current_size = len(idxs)

        if current_size > target_size:
            # UNDERSAMPLING
            new_idxs = random.sample(idxs, target_size)

        elif current_size < target_size:
            # OVERSAMPLING
            extra = random.choices(idxs, k=target_size - current_size)
            new_idxs = idxs + extra

        else:
            new_idxs = idxs

        balanced_groups[action_name] = dataset.select(new_idxs)

    balanced_dataset = DatasetDict(balanced_groups)

    merged_dataset = concatenate_datasets(list(balanced_dataset.values()))
    merged_dataset = merged_dataset.shuffle(seed=42)
    return merged_dataset

model_name = os.getenv('MODEL_NAME', "Qwen3-4B-Instruct-2507")
model_path = f'models/{model_name}'
dataset_dir = os.getenv('DATASET_DIR', 'data')
dataset_size = int(os.getenv('DATASET_SIZE', 100))
print(f'== Model path: {model_path}\n')
print(f'== Dataset dir: {dataset_dir}\n')
num_train_epoch = int(os.getenv('NUM_TRAIN_EPOCH', 1))
print(f'== Number of training epoch: {num_train_epoch}\n')

early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=2,
    early_stopping_threshold=0.0
)

tokenizer = AutoTokenizer.from_pretrained(model_path)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"pad_token set as eos_token: {tokenizer.eos_token}")

tokenizer.padding_side = "right"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
)

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

model.enable_input_require_grads()
model.config.use_cache = False
model.print_trainable_parameters()

dataset = normalize_dataset(f'{dataset_dir}/*.jsonl', dataset_size)

def format_example(example):
    messages = example["messages"]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return {"text": text}

dataset = dataset.map(
    format_example,
    remove_columns=dataset.column_names,
)

def analyze_token_lengths():
    lengths = []
    for example in dataset:
        tokens = tokenizer.encode(example["text"], add_special_tokens=False)
        lengths.append(len(tokens))
    max_len = max(lengths)
    return max_len

max_size = analyze_token_lengths()
max_size = max(max_size, 1024)

print(f'==> SFTConfig. max_seq_length: {max_size}')

response_template = "<|im_start|>assistant\n"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

sft_config = SFTConfig(
    output_dir=f"outputs/{model_name}",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    num_train_epochs=num_train_epoch,
    logging_steps=5,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=False,
    bf16=True,
    dataset_text_field="text",
    max_seq_length=max_size,
    packing=False,
    half_precision_backend="no",
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    report_to="none",
    eval_steps=200,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    #eval_dataset=dataset["validation"],  # на "потом"
    data_collator=collator,
    args=sft_config,
)

trainer.train()

save_dir = f"outputs/{model_name}/final_adapter"
model.save_pretrained(save_dir, safe_serialization=False)
tokenizer.save_pretrained(save_dir)
