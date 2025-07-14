#!/usr/bin/env python
"""Fine-tune a base Llama model using LoRA adapters."""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import List

import yaml
from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

try:
    import wandb
except Exception:  # pragma: no cover - wandb is optional
    wandb = None

LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")


def setup_logging() -> logging.Logger:
    """Configure a logger that writes to the shared logs directory."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    logger = logging.getLogger("train_lora")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(os.path.join(LOGS_DIR, "train_lora.log"))
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def load_config(path: str) -> dict:
    """Load a YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_datasets(paths: List[str], logger: logging.Logger):
    """Load and concatenate multiple JSONL datasets."""
    datasets = []
    for p in paths:
        logger.info("Loading dataset %s", p)
        ds = load_dataset("json", data_files=p, split="train")
        datasets.append(ds)
    if len(datasets) > 1:
        logger.info("Concatenating %d datasets", len(datasets))
        return concatenate_datasets(datasets)
    return datasets[0]


def tokenize_dataset(dataset, tokenizer):
    """Tokenize dataset entries for causal language modeling."""

    def _tokenize(example):
        if isinstance(example.get("messages"), list):
            if getattr(tokenizer, "apply_chat_template", None):
                text = tokenizer.apply_chat_template(
                    example["messages"], tokenize=False, add_generation_prompt=False
                )
            else:
                text = "\n".join(m.get("content", "") for m in example["messages"])
        else:
            text = example.get("text") or json.dumps(example)
        tokens = tokenizer(text)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    return dataset.map(_tokenize, remove_columns=dataset.column_names)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a model with LoRA")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    logger = setup_logging()
    cfg = load_config(args.config)

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    wb_cfg = cfg.get("wandb", {})

    if wb_cfg.get("enabled") and wandb is not None:
        wandb.init(project=wb_cfg.get("project", "llama-finetune"), name=wb_cfg.get("run_name"))
        wandb.config.update(cfg)

    token = model_cfg.get("hf_token") or os.environ.get("HF_TOKEN")

    logger.info("Loading base model from %s", model_cfg["base_model_path"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["base_model_path"], use_fast=False, token=token, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model_path"],
        device_map="auto",
        torch_dtype="auto",
        token=token,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    logger.info("Applying LoRA configuration")
    lora_config = LoraConfig(
        r=train_cfg.get("lora_r", 8),
        lora_alpha=train_cfg.get("lora_alpha", 32),
        lora_dropout=train_cfg.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_datasets(train_cfg.get("datasets", []), logger)
    dataset = tokenize_dataset(dataset, tokenizer)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=train_cfg.get("output_dir", "checkpoints"),
        num_train_epochs=train_cfg.get("num_epochs", 1),
        per_device_train_batch_size=train_cfg.get("batch_size", 1),
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 100),
        report_to="wandb" if wb_cfg.get("enabled") and wandb is not None else "none",
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=dataset, data_collator=data_collator)

    logger.info("Starting training")
    trainer.train()

    os.makedirs(train_cfg.get("output_dir", "checkpoints"), exist_ok=True)
    model.save_pretrained(train_cfg.get("output_dir"))
    tokenizer.save_pretrained(train_cfg.get("output_dir"))
    logger.info("Training complete. Adapters saved to %s", train_cfg.get("output_dir"))

    if wb_cfg.get("enabled") and wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
