#!/usr/bin/env python
"""Merge a LoRA adapter into a base model."""

import argparse
import logging
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")


def setup_logging() -> logging.Logger:
    """Configure and return a module logger."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    logger = logging.getLogger("merge_adapters")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(os.path.join(LOGS_DIR, "merge_adapters.log"))
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def main() -> None:
    """Entry point for the adapter merge CLI."""
    parser = argparse.ArgumentParser(description="Merge LoRA adapters into a base model")
    parser.add_argument("base_model", help="Path to the base model directory")
    parser.add_argument("adapter", help="Path to the LoRA adapter directory")
    parser.add_argument("output_dir", help="Where to save the merged model")
    parser.add_argument("--token", dest="token", default=None, help="Hugging Face access token")
    args = parser.parse_args()

    logger = setup_logging()

    token = args.token or os.environ.get("HF_TOKEN")

    logger.info("Loading base model from %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=False,
        token=token,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype="auto",
        token=token,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading LoRA adapter from %s", args.adapter)
    model = PeftModel.from_pretrained(model, args.adapter)

    logger.info("Merging adapter and unloading")
    model = model.merge_and_unload()

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("Saving merged model to %s", args.output_dir)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Merge complete")


if __name__ == "__main__":
    main()
