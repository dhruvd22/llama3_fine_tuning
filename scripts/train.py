#!/usr/bin/env python
"""Train a model using Axolotl with W&B logging."""
import argparse
import logging
import os
import subprocess

LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")


def setup_logging() -> logging.Logger:
    """Configure and return a module logger."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.FileHandler(os.path.join(LOGS_DIR, "train.log"))
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    return logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Axolotl training")
    parser.add_argument("--config", required=True, help="Path to Axolotl YAML config")
    args = parser.parse_args()

    logger = setup_logging()

    wandb_key = os.environ.get("WANDB_API_KEY")
    if not wandb_key:
        logger.error("WANDB_API_KEY environment variable is not set")
        raise SystemExit("WANDB_API_KEY must be provided for W&B logging")

    cmd = ["axolotl", args.config]
    logger.info("Executing: %s", " ".join(cmd))

    result = subprocess.run(cmd, env=os.environ)
    if result.returncode != 0:
        logger.error("Training failed with code %s", result.returncode)
        raise SystemExit(result.returncode)

    logger.info("Training finished successfully")


if __name__ == "__main__":
    main()
