#!/usr/bin/env python
"""Preprocess NL/SQL datasets using prompt templates."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import yaml
from jinja2 import Template

LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")


def setup_logging() -> logging.Logger:
    """Configure and return a module logger."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    logger = logging.getLogger("preprocess_datasets")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.FileHandler(os.path.join(LOGS_DIR, "preprocess_datasets.log"))
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    return logger


@dataclass
class DatasetConfig:
    """Configuration for a single dataset to preprocess."""

    input: str
    output: str
    template: str | None = None


@dataclass
class PreprocessConfig:
    """Configuration loaded from the YAML config file."""

    default_template: str
    datasets: List[DatasetConfig]

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "PreprocessConfig":
        datasets = [DatasetConfig(**ds) for ds in cfg.get("datasets", [])]
        return PreprocessConfig(cfg["default_template"], datasets)


def load_config(path: str) -> PreprocessConfig:
    """Load preprocessing configuration from a YAML file."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return PreprocessConfig.from_dict(cfg)


def load_template(path: str) -> Template:
    """Load a JSON template file and return a Jinja2 template."""
    with open(path, "r") as f:
        template_json = json.load(f)
    # The template is stored as JSON; convert to string for Jinja2
    return Template(json.dumps(template_json))


def read_dataset(path: str) -> List[Dict[str, Any]]:
    """Load the dataset from JSON. Supports a list or JSON lines."""
    with open(path, "r") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            return json.load(f)
        # Treat as JSON Lines
        return [json.loads(line) for line in f if line.strip()]


def render_entries(data: Iterable[Dict[str, Any]], template: Template) -> List[str]:
    """Render each dataset entry with the template."""
    rendered: List[str] = []
    for item in data:
        sql_value = (
            item.get("sql")
            or item.get("POSTGRESQL_QUERY")
            or item.get("answer")
            or item.get("query")
        )
        answer_value = (
            item.get("answer")
            or item.get("sql")
            or item.get("POSTGRESQL_QUERY")
            or item.get("query")
        )
        context = {
            "question": item.get("question")
            or item.get("NATURAL_LANGUAGE_QUESTION")
            or item.get("natural_language_question"),
            "sql": sql_value,
            "answer": answer_value,
            "schema": item.get("schema")
            or item.get("SCHEMA_JSON")
            or item.get("schema_json")
            or "",
        }
        rendered.append(template.render(**context))
    return rendered


def write_output(lines: Iterable[str], path: str) -> None:
    """Write rendered JSON objects, one per line."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for line in lines:
            f.write(line.rstrip() + "\n")


def process_dataset(cfg: DatasetConfig, default_template: Template, logger: logging.Logger) -> None:
    template = default_template
    if cfg.template:
        logger.info("Using custom template %s", cfg.template)
        template = load_template(cfg.template)

    logger.info("Reading dataset from %s", cfg.input)
    data = read_dataset(cfg.input)
    logger.info("Loaded %d records", len(data))

    rendered = render_entries(data, template)

    logger.info("Writing preprocessed data to %s", cfg.output)
    write_output(rendered, cfg.output)
    logger.info("Finished processing %s", cfg.input)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess NL/SQL datasets")
    parser.add_argument("--config", required=True, help="YAML config file")
    args = parser.parse_args()

    logger = setup_logging()
    cfg = load_config(args.config)

    default_template = load_template(cfg.default_template)

    for ds_cfg in cfg.datasets:
        process_dataset(ds_cfg, default_template, logger)


if __name__ == "__main__":
    main()
