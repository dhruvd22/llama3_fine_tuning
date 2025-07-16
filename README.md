# Llama 3.1 Fine-Tuning Framework

This repository contains a lightweight setup for fine-tuning and running Llama 3 based models on GPU machines such as [RunPod](https://runpod.io/).
Training and experiment management are handled by standalone Python scripts built on the Hugging Face Transformers library, with optional Weights & Biases integration for experiment tracking.

## Repository Layout

- `configs/` – YAML configuration files for training, inference, and preprocessing
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

For LoRA fine-tuning you can use the provided `train_lora.py` script. The default
configuration sets the LoRA rank to **16** and accumulates gradients over four
steps. W&B integration reads the API key from the `WANDB_API_KEY` environment
variable.

```bash
export HF_TOKEN=<your huggingface token>
export WANDB_API_KEY=<your wandb key>
python scripts/train_lora.py --config configs/train_lora.yaml
```

Training logs are written to `logs/train_lora.log` and the adapters plus the
`training_config.yaml` used for the run are saved under the configured
`output_dir`.


## Preprocess NL/SQL Datasets

Use `scripts/preprocess_datasets.py` to convert training datasets into the prompt format expected by the `train_lora.py` script. The command reads a YAML configuration describing one or more datasets and the prompt templates to apply.

1. Edit `configs/preprocess.yaml` and set the path to your input dataset and desired output location. By default the `default_template` located under `configs/prompt_templates/` will be used.
2. Run the preprocessing script:

```bash
python scripts/preprocess_datasets.py --config configs/preprocess.yaml
```

The script outputs a JSON Lines file where each line represents a training example rendered with the prompt template. Custom templates can be provided per dataset in the YAML configuration if needed.

## Evaluate Models

`scripts/evaluate.py` computes an accuracy score for one or more models using a dataset of question/answer pairs.

1. Edit `configs/evaluate.yaml` to list the models to test, the evaluation dataset path and the prompt template.
2. Run the script:

```bash
python scripts/evaluate.py --config configs/evaluate.yaml
```

Accuracy for each model is printed at the end of the run. A log file named
`evaluate_<model>_<timestamp>.log` is created in the `logs/` directory for each
model, recording the full prompts sent to the model and the responses
generated.
