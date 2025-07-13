#!/usr/bin/env python
"""Download an LLM from Hugging Face Hub."""
import argparse
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Download a model from Hugging Face")
    parser.add_argument("model_id", help="Hugging Face model ID, e.g., meta-llama/Meta-Llama-3-8B")
    parser.add_argument("output_dir", help="Directory to save the model")
    parser.add_argument("--token", dest="token", default=None, help="Hugging Face access token")
    args = parser.parse_args()

    snapshot_download(
        repo_id=args.model_id,
        local_dir=args.output_dir,
        local_dir_use_symlinks=False,
        token=args.token,
    )


if __name__ == "__main__":
    main()
