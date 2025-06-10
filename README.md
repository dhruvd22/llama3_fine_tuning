# Llama-3-8B Text-to-SQL Fine-Tuning
Fine-tuning Llama-3-8B for Text-to-SQL (RunPod + Hydra + W&amp;B)
Scripts, configs, and experiments for fine-tuning Meta AI Llama-3-8B into a Text-to-SQL generator.

## Folder layout
- `datasets/`  → raw & processed Spider, WikiSQL, BIRD, …
- `configs/`    → Axolotl YAML config templates
- `scripts/`    → reusable shell / Python helpers
- `checkpoints/`→ LoRA & merged weights
- `models/`     → final packaged models
- `logs/`       → training & evaluation logs

## Requirements
The helper scripts assume common utilities like `wget`, `tar`, and `unzip` are
installed. On Debian/Ubuntu systems you can install them with:

```bash
sudo apt-get update && sudo apt-get install -y wget tar unzip
```

