# Llama 3.1 Fine-Tuning Framework

This repository contains a lightweight setup for fine-tuning and running Llama 3 based models on GPU machines such as [RunPod](https://runpod.io/).  
Training and experiment management is handled via the [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) framework with optional Weights & Biases integration.

## Repository Layout

- `configs/` – YAML configuration files for Axolotl and inference
- `scripts/` – utility scripts for downloading models and running inference
- `datasets/` – place your training datasets here
- `checkpoints/` – fine‑tuning outputs and LoRA adapters
- `models/` – downloaded or merged models
- `logs/` – output logs from all scripts

Each directory contains a `.gitkeep` file so it is tracked even if empty.

## Quick Start

1. **Download a base model**

   ```bash
   export HF_TOKEN=<your huggingface token>
   python scripts/download_model.py meta-llama/Llama-3.1-8B models/my-llama
   ```

2. **Run inference**

   Edit `configs/inference.yaml` to point `base_model_path` to your model directory then run:

   ```bash
   python scripts/inference.py --config configs/inference.yaml --prompt "Hello"
   ```

For training, create an Axolotl configuration under `configs/` and launch it inside your RunPod container. Training metrics can be logged to W&B by setting the appropriate options in your Axolotl config.

