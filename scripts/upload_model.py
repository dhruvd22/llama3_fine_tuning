#!/usr/bin/env python
"""Upload model directories to the Hugging Face Hub using the CLI."""
import argparse
import logging
import os
import subprocess

LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")


def setup_logging() -> logging.Logger:
    """Configure and return a module logger."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    logger = logging.getLogger("upload_model")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(os.path.join(LOGS_DIR, "upload_model.log"))
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def main() -> None:
    """Entry point for the upload CLI."""
    parser = argparse.ArgumentParser(description="Upload model files to the Hugging Face Hub")
    parser.add_argument("local_directory", help="Path to the local directory to upload")
    parser.add_argument(
        "upload_directory",
        help="Destination path in the repo, e.g. 'base_3_1_8b/'",
    )
    parser.add_argument(
        "--repo_id",
        default="dhruvdahiya/my_llama_models",
        help="Repository on the Hub to upload to",
    )
    args = parser.parse_args()

    logger = setup_logging()
    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.error("HF_TOKEN environment variable is not set")
        raise SystemExit("HF_TOKEN must be provided for authentication")

    command = [
        "huggingface-cli",
        "upload",
        args.repo_id,
        args.local_directory,
        args.upload_directory,
        "--repo-type",
        "model",
        "--token",
        token,
    ]

    logger.info("Running: %s", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Upload failed: %s", result.stderr.strip())
        raise SystemExit(result.returncode)

    logger.info("Upload complete")
    if result.stdout:
        logger.info(result.stdout.strip())


if __name__ == "__main__":
    main()
