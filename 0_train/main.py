import json
from collections import defaultdict
import random
from typing import Dict, List, Tuple
import numpy as np

import torch
import os
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from sklearn.metrics import accuracy_score, f1_score, classification_report
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ValidationMetricsCallback(TrainerCallback):
    """Callback для вычисления метрик на валидационном датасете"""

    def __init__(self, trainer, eval_dataset, tokenizer):
        self.trainer = trainer
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.best_f1 = 0.0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Вызывается после каждой валидации"""
        if metrics and 'eval_loss' in metrics:
            logger.info(f"Validation Loss: {metrics['eval_loss']:.4f}")

            # Вычисляем дополнительные метрики
            custom_metrics = self.compute_custom_metrics()
            metrics.update(custom_metrics)

            # Сохраняем лучшую модель по F1
            if custom_metrics.get('eval_f1', 0) > self.best_f1:
                self.best_f1 = custom_metrics['eval_f1']
                logger.info(f"New best F1 score: {self.best_f1:.4f}")

    def compute_custom_metrics(self) -> Dict[str, float]:
        """Вычисление метрик классификации"""
        # Здесь можно добавить кастомную логику оценки
        # Например, точность предсказания имени action
        return {}


def normalize_dataset(
        data_files: str,
        dataset_group_size: int = -1,
        validation_split: float = 0.15,
        test_split: float = 0.0
) -> Tuple[Dataset, Dataset, Dataset]:
    logger.info(f"Loading dataset from: {data_files}")
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
    logger.info(f"Original class distribution: {sizes}")

    target_size = dataset_group_size if dataset_group_size != -1 else int(sum(sizes.values()) / len(sizes))
    logger.info(f"Target size per class: {target_size}")

    train_groups = {}
    val_groups = {}
    test_groups = {}

    for action_name, idxs in indices_by_action.items():
        random.shuffle(idxs)

        current_size = len(idxs)

        n_test = max(0, int(current_size * test_split))
        n_val = max(1, int(current_size * validation_split))
        n_train = current_size - n_test - n_val

        test_idxs = idxs[:n_test]
        val_idxs = idxs[n_test:n_test + n_val]
        train_idxs = idxs[n_test + n_val:]

        if len(train_idxs) > target_size:
            # UNDERSAMPLING
            train_idxs = random.sample(train_idxs, target_size)
        elif len(train_idxs) < target_size:
            # OVERSAMPLING
            extra = random.choices(train_idxs, k=target_size - len(train_idxs))
            train_idxs = train_idxs + extra

        train_groups[action_name] = dataset.select(train_idxs)
        val_groups[action_name] = dataset.select(val_idxs)
        test_groups[action_name] = dataset.select(test_idxs)

    train_dataset = concatenate_datasets(list(train_groups.values())).shuffle(seed=42)
    val_dataset = concatenate_datasets(list(val_groups.values())).shuffle(seed=42)
    test_dataset = concatenate_datasets(list(test_groups.values())).shuffle(seed=42)

    logger.info(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset


def compute_metrics_for_evaluation(
        model: PeftModel,
        eval_dataset: Dataset,
        tokenizer: AutoTokenizer,
        device: str = "cuda"
) -> Dict[str, float]:
    model.eval()
    predictions = []
    references = []

    logger.info("Computing evaluation metrics...")

    with torch.no_grad():
        for example in eval_dataset:
            true_action = example["action_name"]

            messages = example["messages"][:-1]

            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            try:
                start_idx = generated_text.rfind("{")
                end_idx = generated_text.rfind("}") + 1

                if start_idx != -1 and end_idx > start_idx:
                    json_str = generated_text[start_idx:end_idx]
                    predicted_data = json.loads(json_str)
                    predicted_action = predicted_data.get("action", {}).get("name", "unknown")
                else:
                    predicted_action = "unknown"
            except:
                predicted_action = "unknown"

            predictions.append(predicted_action)
            references.append(true_action)

    accuracy = accuracy_score(references, predictions)
    f1_macro = f1_score(references, predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(references, predictions, average='weighted', zero_division=0)

    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"F1 Score (Macro): {f1_macro:.4f}")
    logger.info(f"F1 Score (Weighted): {f1_weighted:.4f}")
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(references, predictions))
    logger.info("=" * 50 + "\n")

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


def analyze_token_lengths(dataset: Dataset, tokenizer: AutoTokenizer) -> int:
    lengths = []
    for example in dataset:
        tokens = tokenizer.encode(example["text"], add_special_tokens=False)
        lengths.append(len(tokens))

    def round_up_to_even(x):
        return x if x % 2 == 0 else x + 1

    max_len = max(lengths)
    max_len = round_up_to_even(max_len)

    recommended_length = max(max_len, 1024)
    logger.info(f"Recommended max_seq_length: {recommended_length}")

    return recommended_length


def main():
    model_name = os.getenv('MODEL_NAME', "Qwen3-4B-Instruct-2507")
    model_path = f'models/{model_name}'
    dataset_dir = os.getenv('DATASET_DIR', 'data')
    dataset_size = int(os.getenv('DATASET_SIZE', 100))
    num_train_epoch = int(os.getenv('NUM_TRAIN_EPOCH', 3))

    validation_split = float(os.getenv('VALIDATION_SPLIT', 0.15))
    learning_rate = float(os.getenv('LEARNING_RATE', 5e-5))
    batch_size = int(os.getenv('BATCH_SIZE', 4))
    gradient_accumulation = int(os.getenv('GRADIENT_ACCUMULATION', 4))

    lora_rank = int(os.getenv('LORA_RANK', 64))
    lora_alpha = int(os.getenv('LORA_ALPHA', 128))

    use_eval = os.getenv('USE_EVAL', 'false').lower() in ("1", "true", "yes", "on")

    logger.info("=" * 50)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 50)
    logger.info(f"Model: {model_name}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Dataset dir: {dataset_dir}")
    logger.info(f"Target dataset size per class: {dataset_size}")
    logger.info(f"Epochs: {num_train_epoch}")
    logger.info(f"Validation split: {validation_split}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Gradient accumulation: {gradient_accumulation}")
    logger.info("=" * 50 + "\n")

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Pad token set as eos_token: {tokenizer.eos_token}")

    tokenizer.padding_side = "right"

    logger.info("Loading model with 4-bit quantization...")
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

    logger.info("Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
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

    train_dataset, val_dataset, test_dataset = normalize_dataset(
        f'{dataset_dir}/*.jsonl',
        dataset_size,
        validation_split=validation_split
    )

    def format_example(example):
        messages = example["messages"]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    train_dataset = train_dataset.map(format_example, remove_columns=train_dataset.column_names)
    val_dataset_formatted = val_dataset.map(format_example, remove_columns=val_dataset.column_names)
    test_dataset_formatted = test_dataset.map(format_example, remove_columns=test_dataset.column_names)

    max_seq_length = analyze_token_lengths(train_dataset, tokenizer)

    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.001
    )

    sft_config = SFTConfig(
        output_dir=f"outputs/{model_name}",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        num_train_epochs=num_train_epoch,

        logging_steps=10,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,

        eval_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        max_grad_norm=1.0,

        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,

        report_to="tensorboard",
    )

    logger.info("Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset_formatted,
        data_collator=collator,
        args=sft_config,
        callbacks=[early_stopping_callback],
    )

    logger.info("Starting training...")
    trainer.train()

    if use_eval:
        logger.info("\nEvaluating on test set...")
        final_metrics = compute_metrics_for_evaluation(
            model=model,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            device=model.device
        )
        metrics_path = f"outputs/{model_name}/final_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        logger.info(f"Metrics saved to: {metrics_path}")

    save_dir = f"outputs/{model_name}/final_adapter"
    logger.info(f"Saving model to: {save_dir}")
    model.save_pretrained(save_dir, safe_serialization=False)
    tokenizer.save_pretrained(save_dir)

    logger.info("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
