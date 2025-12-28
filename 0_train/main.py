import torch
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

model_name = os.getenv('MODEL_NAME', "Qwen3-4B-Instruct-2507")
model_path = f'models/{model_name}'
dataset_file_name = os.getenv('DATASET_FILE_NAME', "4_training_dataset.jsonl")
validation_dataset_file_name = os.getenv('VALIDATION_DATASET_FILE_NAME', "4_validation_dataset.jsonl")
print(f'== Model path: {model_path}\n')
print(f'== Dataset: {dataset_file_name}\n')
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
#tokenizer.pad_token = tokenizer.eos_token

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

dataset = load_dataset(
    "json",
    data_files={
        "train": f"data/{dataset_file_name}",
        #"validation": f"data/{validation_dataset_file_name}", # на "потом"
    }
)

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
    remove_columns=dataset["train"].column_names,
)


def analyze_token_lengths():
    lengths = []
    for example in dataset["train"]:
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
    learning_rate=2e-4,
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
    train_dataset=dataset["train"],
    #eval_dataset=dataset["validation"],  # на "потом"
    data_collator=collator,
    args=sft_config,
)

trainer.train()

save_dir = f"outputs/{model_name}/final_adapter"
model.save_pretrained(save_dir, safe_serialization=False)
tokenizer.save_pretrained(save_dir)
