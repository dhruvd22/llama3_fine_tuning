#!/usr/bin/env python
"""Load a model from config and run a single prompt."""
import argparse
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_from_config(cfg_path: str):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg.get('model', {})
    base_path = model_cfg['base_model_path']
    adapters = model_cfg.get('adapters', [])

    tokenizer = AutoTokenizer.from_pretrained(base_path)
    model = AutoModelForCausalLM.from_pretrained(base_path)

    for adapter in adapters:
        model = PeftModel.from_pretrained(model, adapter)

    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Run inference from YAML config")
    parser.add_argument('--config', required=True, help='Path to YAML config')
    parser.add_argument('--prompt', default='Hello', help='Prompt text')
    parser.add_argument('--max_tokens', type=int, default=50)
    args = parser.parse_args()

    model, tokenizer = load_from_config(args.config)
    inputs = tokenizer(args.prompt, return_tensors='pt')
    outputs = model.generate(**inputs, max_new_tokens=args.max_tokens)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == '__main__':
    main()
