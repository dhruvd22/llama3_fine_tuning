#!/usr/bin/env python3
"""Simple interactive REPL for a local Llama 3.1-8B model.

Usage:
    python 01_simple_model_infererence.py --model_dir /path/to/model
"""

import argparse
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an interactive Llama 3.1-8B chat")
    parser.add_argument("--model_dir", required=True, help="Directory containing the model")
    args = parser.parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    dtype = torch.float16 if device != "cpu" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=dtype)
    model.to(device)
    model.eval()

    print("Model loaded. Type 'quit' or 'exit' to stop.")

    while True:
        try:
            text = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if text.lower() in {"quit", "exit"}:
            break
        if not text:
            continue

        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        try:
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                )
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("RuntimeError: CUDA out of memory", file=sys.stderr)
                break
            raise

        result = tokenizer.decode(output[0], skip_special_tokens=True)
        result = result[len(text) :].strip()
        print(f"Llama: {result}")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        pass
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            pass
        else:
            raise

