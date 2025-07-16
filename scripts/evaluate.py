#!/usr/bin/env python
"""Evaluate one or more models using a question/answer dataset."""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Tuple
from datetime import datetime

import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from jinja2 import Environment, meta

LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")


def setup_logging() -> logging.Logger:
    """Configure and return a logger for the evaluation script."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    logger = logging.getLogger("evaluate")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.FileHandler(os.path.join(LOGS_DIR, "evaluate.log"))
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    return logger


def create_model_logger(name: str) -> Tuple[logging.Logger, str]:
    """Return a logger writing to a model-specific log file."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOGS_DIR, f"evaluate_{safe_name}_{timestamp}.log")
    logger = logging.getLogger(f"evaluate.{safe_name}.{timestamp}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(fmt)
    logger.handlers = []
    logger.addHandler(handler)
    return logger, log_path


def load_prompt_template(path: str):
    """Return a Jinja2 template and the variables it requires."""
    with open(path, "r") as f:
        data = json.load(f)
    template_str = json.dumps(data)
    env = Environment()
    ast = env.parse(template_str)
    variables = sorted(meta.find_undeclared_variables(ast))
    template = env.from_string(template_str)
    return template, variables


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load evaluation data from JSON or JSON Lines."""
    with open(path, "r") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def load_model(cfg: Dict[str, Any], logger: logging.Logger):
    """Load a base model and optional adapters based on the configuration."""
    base_path = cfg["base_model_path"]
    token = cfg.get("hf_token") or os.environ.get("HF_TOKEN")

    logger.info("Loading base model from %s", base_path)
    tokenizer = AutoTokenizer.from_pretrained(
        base_path,
        use_fast=False,
        token=token,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        token=token,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    for adapter in cfg.get("adapters", []):
        logger.info("Applying adapter %s", adapter)
        model = PeftModel.from_pretrained(model, adapter)

    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_tokens: int) -> str:
    """Generate a model response for the provided prompt."""
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    gen_tokens = outputs[0, inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True)


def evaluate_model(
    model,
    tokenizer,
    dataset: List[Dict[str, Any]],
    template,
    variables: List[str],
    max_tokens: int,
    logger: logging.Logger,
) -> float:
    """Return the accuracy of the model on the dataset."""
    correct = 0
    for item in dataset:
        context = {var: item.get(var, "") for var in variables}
        rendered = template.render(**context)
        messages = json.loads(rendered)["messages"]
        if getattr(tokenizer, "chat_template", None):
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt_text = "\n".join(m.get("content", "") for m in messages)

        logger.info("Prompting with question: %s", item.get("question"))
        logger.info("Full prompt sent to model:\n%s", prompt_text)
        prediction = generate(model, tokenizer, prompt_text, max_tokens).strip()
        expected = (item.get("answer") or item.get("sql") or "").strip()
        if prediction == expected:
            correct += 1
        logger.info("Model response: %s", prediction)
        logger.info("Expected: %s", expected)

    return correct / len(dataset) if dataset else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate models from a YAML config")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--max_tokens", type=int, default=50)
    args = parser.parse_args()

    logger = setup_logging()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    data_path = cfg.get("data_path")
    if not data_path:
        raise ValueError("data_path must be specified in the config file")
    dataset = load_dataset(data_path)

    template_path = cfg.get("prompt_template")
    if not template_path:
        raise ValueError("prompt_template must be specified in the config file")
    template, variables = load_prompt_template(template_path)

    results: List[Tuple[str, float]] = []
    for model_cfg in cfg.get("models", []):
        name = model_cfg.get("name") or model_cfg.get("base_model_path")
        logger.info("Evaluating model: %s", name)
        model, tokenizer = load_model(model_cfg, logger)
        run_logger, path = create_model_logger(name)
        logger.info("Logging prompts and responses to %s", path)
        accuracy = evaluate_model(
            model,
            tokenizer,
            dataset,
            template,
            variables,
            args.max_tokens,
            run_logger,
        )
        results.append((name, accuracy))
        logger.info("Accuracy for %s: %.4f", name, accuracy)

    print("\n=== Evaluation Results ===")
    for name, acc in results:
        print(f"{name}: {acc:.4f}")


if __name__ == "__main__":
    main()
