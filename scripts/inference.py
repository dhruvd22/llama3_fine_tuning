#!/usr/bin/env python
"""Load a model from config and run prompt-based inference."""
import argparse
import json
import logging
import os
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
from jinja2 import Environment, meta


class StopOnSequences(StoppingCriteria):
    """Stop text generation once any of the stop sequences is produced."""

    def __init__(self, stop_sequences, tokenizer):
        super().__init__()
        self.stop_sequences = stop_sequences
        self.tokenizer = tokenizer
        self.generated = ""

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:  # type: ignore[override]
        # Decode only the newly generated token and accumulate it
        token = input_ids[0, -1].item()
        self.generated += self.tokenizer.decode([token], skip_special_tokens=False)
        for seq in self.stop_sequences:
            if self.generated.endswith(seq):
                return True
        return False

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


def load_from_config(cfg_path: str, logger: logging.Logger):
    """Load a model, tokenizer and prompt template based on a YAML configuration."""
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

    template_path = cfg.get("prompt_template")
    if not template_path:
        raise ValueError("prompt_template must be specified in the config file")
    template, variables = load_prompt_template(template_path)

    stop_sequences = cfg.get("stop", [])

    model.eval()
    logger.info("Model ready for inference")
    return model, tokenizer, template, variables, stop_sequences


def main() -> None:
    """Run inference using the provided configuration file."""
    parser = argparse.ArgumentParser(description="Run inference from YAML config")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--max_tokens", type=int, default=50)
    args = parser.parse_args()

    logger = setup_logging()

    model, tokenizer, template, variables, stop_sequences = load_from_config(args.config, logger)

    def generate_response(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        stopping = None
        if stop_sequences:
            stopping = StoppingCriteriaList([StopOnSequences(stop_sequences, tokenizer)])
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            stopping_criteria=stopping,
        )
        generated_tokens = outputs[0, inputs["input_ids"].shape[-1]:]
        text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        if stop_sequences:
            for seq in stop_sequences:
                if seq in text:
                    text = text.split(seq)[0]
                    break
        return text

    while True:
        values = {}
        for var in variables:
            values[var] = input(f"Enter {var}: ").strip()

        if any(v.lower() in {"quit", "exit"} for v in values.values()):
            print("Exiting.")
            break

        rendered = template.render(**values)
        messages = json.loads(rendered)["messages"]
        # If the tokenizer defines a chat template, use it to build the prompt.
        # Base models might not provide this attribute so we fall back to a
        # simple concatenation of message contents.
        if getattr(tokenizer, "chat_template", None):
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = "\n".join(m.get("content", "") for m in messages)

        logger.info("Generating response for values: %s", values)
        response = generate_response(prompt_text)
        logger.info("Model output: %s", response)

        print("\n=== Prompt Sent to Model ===")
        print(prompt_text)
        print("\n=== Model Response ===")
        print(response)



if __name__ == '__main__':
    main()
