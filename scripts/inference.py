#!/usr/bin/env python
"""Load a model from config and run a single prompt."""
import argparse
import logging
import os
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Log files are stored in the repository's shared logs directory
LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")


def setup_logging() -> logging.Logger:
    """Configure and return a module logger."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    logger = logging.getLogger("inference")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.FileHandler(os.path.join(LOGS_DIR, "inference.log"))
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    return logger

def load_from_config(cfg_path: str, logger: logging.Logger):
    """Load a model and tokenizer based on a YAML configuration."""
    # Read the YAML configuration describing the model and optional adapters
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg.get("model", {})
    base_path = model_cfg["base_model_path"]
    adapters = model_cfg.get("adapters", [])
    token = model_cfg.get("hf_token") or os.environ.get("HF_TOKEN")

    logger.info("Loading base model from %s", base_path)
    tokenizer = AutoTokenizer.from_pretrained(
        base_path,
        token=token,
        use_fast=False,
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

    for adapter in adapters:
        logger.info("Applying adapter %s", adapter)
        model = PeftModel.from_pretrained(model, adapter)

    model.eval()
    logger.info("Model ready for inference")
    return model, tokenizer


def main() -> None:
    """Run inference using the provided configuration file."""
    parser = argparse.ArgumentParser(description="Run inference from YAML config")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--prompt", default=None, help="Prompt text; if not provided you will be asked for it")
    parser.add_argument("--max_tokens", type=int, default=50)
    args = parser.parse_args()

    logger = setup_logging()

    model, tokenizer = load_from_config(args.config, logger)

    def generate_response(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model.generate(**inputs, max_new_tokens=args.max_tokens)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    prompt = args.prompt
    if prompt is None:
        prompt = input("Enter prompt: ").strip()

    while True:
        if prompt.lower() in {"quit", "exit"}:
            print("Exiting.")
            break
        logger.info("Generating response for prompt: %s", prompt)
        response = generate_response(prompt)
        logger.info("Model output: %s", response)
        print(response)
        prompt = input("Enter prompt: ").strip()

if __name__ == '__main__':
    main()
