# Llama 3.1 Fine-Tuning Framework

This repository contains a lightweight setup for fine-tuning and running Llama 3 based models on GPU machines such as [RunPod](https://runpod.io/).  
Training and experiment management is handled via the [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) framework with optional Weights & Biases integration.

## Repository Layout

- `configs/` – YAML configuration files for Axolotl and inference
- `scripts/` – utility scripts for downloading models, running inference, and uploading to the Hugging Face Hub
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

   `configs/inference.yaml` now also defines a `prompt_template` used to build chat prompts. The script will ask for each variable referenced in the template before sending the request to the model. Run the script and it will keep prompting for new values until you type `exit` or `quit`:

    ```bash
    python scripts/inference.py --config configs/inference.yaml
    ```

3. **Upload a model directory to the Hub**

    ```bash
    python scripts/upload_model.py models/my-llama "my-model/"
    ```

For training, create an Axolotl configuration under `configs/` and launch it inside your RunPod container. Training metrics can be logged to W&B by setting the appropriate options in your Axolotl config.


## Preprocess NL/SQL Datasets

Use `scripts/preprocess_datasets.py` to convert training datasets into the prompt format expected by Axolotl. The command reads a YAML configuration describing one or more datasets and the prompt templates to apply.

1. Edit `configs/preprocess.yaml` and set the path to your input dataset and desired output location. By default the `default_template` located under `configs/prompt_templates/` will be used.
2. Run the preprocessing script:

```bash
python scripts/preprocess_datasets.py --config configs/preprocess.yaml
```

The script outputs a JSON Lines file where each line represents a training example rendered with the prompt template. Custom templates can be provided per dataset in the YAML configuration if needed.
