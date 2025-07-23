#!/usr/bin/env python
"""Fine-tune a base Llama model using LoRA adapters."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from typing import List

import yaml
from datasets import (
    Dataset,
    Features,
    List as HFList,
    Sequence,
    Value,
    concatenate_datasets,
    load_dataset,
)
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
PROMPTS_LOG = os.path.join(LOGS_DIR, "train_lora_prompts.log")

# Explicit schema for chat style datasets
MESSAGE_FEATURES = Features(
    {
        "messages": HFList({"role": Value("string"), "content": Value("string")})
    }
)


def setup_logging() -> logging.Logger:
    """Configure a logger that writes to the shared logs directory."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    logger = logging.getLogger("train_lora")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        file_handler = logging.FileHandler(os.path.join(LOGS_DIR, "train_lora.log"))
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)
    return logger


def load_config(path: str) -> dict:
    """Load a YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _parse_jsonlines_file(path: str, logger: logging.Logger) -> Dataset:
    """Read a JSON Lines file and return a Dataset with validated messages."""
    records = []
    with open(path, "r") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as je:
                logger.warning(
                    "Skipping malformed JSON line %d in %s: %s",
                    line_no,
                    path,
                    je,
                )
                continue

            # allow lines formatted either as an object with a "messages" key or
            # directly as a list of message dicts
            if isinstance(record, list):
                record = {"messages": record}
            if not isinstance(record, dict):
                logger.warning(
                    "Skipping non-object JSON line %d in %s", line_no, path
                )
                continue

            msgs = []
            if isinstance(record.get("messages"), list):
                for m in record["messages"]:
                    if not isinstance(m, dict):
                        logger.warning(
                            "Skipping invalid message in line %d of %s", line_no, path
                        )
                        continue
                    msgs.append({"role": m.get("role", ""), "content": m.get("content", "")})
            else:
                logger.warning(
                    "No 'messages' field in line %d of %s", line_no, path
                )
            records.append({"messages": msgs})
    logger.info("Parsed %d valid lines from %s", len(records), path)
    return Dataset.from_list(records, features=MESSAGE_FEATURES)


def load_datasets(paths: List[str], logger: logging.Logger):
    """Load and concatenate multiple JSONL datasets."""
    datasets = []
    for p in paths:
        logger.info("Loading dataset %s", p)
        try:
            ds = load_dataset(
                "json",
                data_files=p,
                split="train",
                features=MESSAGE_FEATURES,
            )
            # enforce message key ordering
            ds = ds.map(
                lambda ex: {
                    "messages": [
                        {"role": m.get("role", ""), "content": m.get("content", "")}
                        for m in (ex.get("messages") or [])
                    ]
                }
            )
        except Exception as e:  # fall back to manual parsing on JSON errors
            logger.error("Failed to load %s with pyarrow: %s", p, e)
            ds = _parse_jsonlines_file(p, logger)
        datasets.append(ds)
    if len(datasets) > 1:
        logger.info("Concatenating %d datasets", len(datasets))
        dataset = concatenate_datasets(datasets)
    else:
        dataset = datasets[0]
    logger.info("Total valid records: %d", dataset.num_rows)
    return dataset


def tokenize_dataset(dataset, tokenizer, prompt_log_path: str | None = None):
    """Tokenize dataset entries for causal language modeling.

    If ``prompt_log_path`` is provided, every prompt text is written to that
    file before tokenization.
    """

    prompt_fh = open(prompt_log_path, "w") if prompt_log_path else None

    def _tokenize(example):
        if isinstance(example.get("messages"), list):
            chat_template = getattr(tokenizer, "chat_template", None)
            if getattr(tokenizer, "apply_chat_template", None) and chat_template:
                text = tokenizer.apply_chat_template(
                    example["messages"], tokenize=False, add_generation_prompt=False
                )
            else:
                text = "\n".join(m.get("content", "") for m in example["messages"])
        else:
            text = example.get("text") or json.dumps(example)
        if prompt_fh:
            prompt_fh.write(text + "\n")
        tokens = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=getattr(tokenizer, "model_max_length", None),
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    try:
        return dataset.map(_tokenize, remove_columns=dataset.column_names)
    finally:
        if prompt_fh:
            prompt_fh.close()


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
        wandb_api_key = os.environ.get("WANDB_API_KEY")
        if wandb_api_key:
            wandb.login(key=wandb_api_key)
        else:
            logger.warning("WANDB_API_KEY environment variable not set")
        wandb.init(
            project=wb_cfg.get("project", "llama-finetune"),
            name=wb_cfg.get("run_name"),
        )
        wandb.config.update(cfg)

    token = model_cfg.get("hf_token") or os.environ.get("HF_TOKEN")

    logger.info("Loading base model from %s", model_cfg["base_model_path"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["base_model_path"],
        use_fast=False,
        token=token,
        trust_remote_code=True,
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
        r=train_cfg.get("lora_r", 16),
        lora_alpha=train_cfg.get("lora_alpha", 32),
        lora_dropout=train_cfg.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_datasets(train_cfg.get("datasets", []), logger)
    dataset = tokenize_dataset(dataset, tokenizer, prompt_log_path=PROMPTS_LOG)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=train_cfg.get("output_dir", "checkpoints"),
        num_train_epochs=train_cfg.get("num_epochs", 1),
        per_device_train_batch_size=train_cfg.get("batch_size", 1),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 100),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        report_to="wandb" if wb_cfg.get("enabled") and wandb is not None else "none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    logger.info("Starting training")
    trainer.train()

    os.makedirs(train_cfg.get("output_dir", "checkpoints"), exist_ok=True)
    model.save_pretrained(train_cfg.get("output_dir"))
    tokenizer.save_pretrained(train_cfg.get("output_dir"))
    shutil.copy2(
        args.config, os.path.join(train_cfg.get("output_dir"), "training_config.yaml")
    )
    logger.info("Training complete. Adapters saved to %s", train_cfg.get("output_dir"))

    if wb_cfg.get("enabled") and wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
