#!/usr/bin/env python
"""Download an LLM from the Hugging Face Hub."""
import argparse
import logging
import os
from huggingface_hub import snapshot_download

LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")


def setup_logging() -> logging.Logger:
    """Configure and return a module logger."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    logger = logging.getLogger("download_model")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.FileHandler(os.path.join(LOGS_DIR, "download_model.log"))
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    return logger


def main() -> None:
    """Entry point for the model download CLI."""
    parser = argparse.ArgumentParser(description="Download a model from Hugging Face")
    parser.add_argument("model_id", help="Hugging Face model ID, e.g., meta-llama/Llama-3.1-8B")
    parser.add_argument("output_dir", help="Directory to save the model")
    parser.add_argument("--token", dest="token", default=None, help="Hugging Face access token")
    args = parser.parse_args()

    logger = setup_logging()

    # Prefer CLI token but fall back to HF_TOKEN environment variable
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        logger.warning(
            "No Hugging Face token provided; download may fail for private models"
        )

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("Downloading %s to %s", args.model_id, args.output_dir)

    snapshot_download(
        repo_id=args.model_id,
        local_dir=args.output_dir,
        local_dir_use_symlinks=False,
        token=token,
    )

    logger.info("Download complete")


if __name__ == "__main__":
    main()
