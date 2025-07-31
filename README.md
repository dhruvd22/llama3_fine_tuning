# Llama 3.1 Fine-Tuning Framework

This repository contains a lightweight setup for fine-tuning and running Llama 3 based models on GPU machines such as [RunPod](https://runpod.io/).
Training and experiment management are handled by standalone Python scripts built on the Hugging Face Transformers library, with optional Weights & Biases integration for experiment tracking.

## Installation

Install the core dependencies with pip:

```bash
pip install torch transformers datasets peft jinja2 pyyaml
```

For SQL validation during evaluation you also need the PostgreSQL driver:

```bash
pip install psycopg2-binary
```

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

`configs/inference.yaml` now also defines a `prompt_template` and optional `stop` sequences used to truncate the model output. The script will ask for each variable referenced in the template before sending the request to the model. Run the script and it will keep prompting for new values until you type `exit` or `quit`:

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

The provided `configs/train_lora.yaml` uses a relatively small batch size to
avoid running out of GPU memory. If you hit CUDA ``OutOfMemory`` errors you can
further decrease ``batch_size`` and increase ``gradient_accumulation_steps`` to
keep a similar effective batch size.

```bash
export HF_TOKEN=<your huggingface token>
export WANDB_API_KEY=<your wandb key>
python scripts/train_lora.py --config configs/train_lora.yaml
```

Training logs are written to `logs/train_lora.log` and the adapters plus the
`training_config.yaml` used for the run are saved under the configured
`output_dir`.

The training script performs a validation pass over each JSONL dataset before
loading. Malformed lines are skipped and every message object is normalised to
contain `role` and `content` fields in a consistent order. This prevents schema
errors when concatenating multiple datasets.

To create a standalone model that includes the fine-tuned weights, merge the
adapters back into the base model:

```bash
python scripts/merge_adapters.py /path/to/base_model /path/to/adapter_dir models/merged-model
```


## Preprocess NL/SQL Datasets

Use `scripts/preprocess_datasets.py` to convert training datasets into the prompt format expected by the `train_lora.py` script. The command reads a YAML configuration describing one or more datasets and the prompt templates to apply.

1. Edit `configs/preprocess.yaml` and set the path to your input dataset and desired output location. By default the `default_template` located under `configs/prompt_templates/` will be used.
2. Run the preprocessing script:

```bash
python scripts/preprocess_datasets.py --config configs/preprocess.yaml
```

The script outputs a JSON Lines file where each line represents a training example rendered with the prompt template. Custom templates can be provided per dataset in the YAML configuration if needed.

## Evaluate Models

`scripts/evaluate.py` validates every SQL statement returned by the model by
executing it on a PostgreSQL database. The final score is a combination of
**50% exact string match** and **50% successful SQL execution**.

1. Configure the database connection using the `DATABASE_URL` environment
   variable or specify the individual `PGHOST`, `PGPORT`, `PGUSER`, `PGPASSWORD`
   and `PGDATABASE` variables. A connection URL can also be provided directly
   with `--db_url`.
2. Edit `configs/evaluate.yaml` to list the models to test, the evaluation
   dataset path and the prompt template.
3. Run the script:

```bash
python scripts/evaluate.py --config configs/evaluate.yaml
```

For each model the script prints the combined score as well as the exact-match
and SQL-validation metrics. A log file named `evaluate_<model>_<timestamp>.log`
is created in the `logs/` directory for each model, recording the full prompts
sent to the model and the generated responses.
