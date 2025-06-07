#!/usr/bin/env bash
set -euo pipefail

ok() { echo -e "\e[32m✔\e[0m $*"; }
err() { echo -e "\e[31m✖\e[0m $*"; }
info() { echo -e "\e[34m➜\e[0m $*"; }

usage() {
  echo "Usage: $0 TARGET_DIR" >&2
  exit 1
}

[[ $# -eq 1 ]] || usage
TARGET_DIR="$1"
HF_REPO="meta-llama/Llama-3.1-8B"

command -v huggingface-cli >/dev/null 2>&1 || {
  err "huggingface-cli not found. Install via 'pip install -U huggingface_hub'."
  exit 1
}

mkdir -p "$TARGET_DIR"

TOKEN_ARG=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  TOKEN_ARG=(--token "$HF_TOKEN")
fi

info "Downloading $HF_REPO to $TARGET_DIR ..."
huggingface-cli download "$HF_REPO" --repo-type model \
  --local-dir "$TARGET_DIR" \
  --resume-download "${TOKEN_ARG[@]}" >/dev/null
ok "Download complete."

info "Running sanity checks ..."
python - <<'PY'
import torch, importlib, sys

if not torch.cuda.is_available():
    print("\u2716 CUDA not available", file=sys.stderr)
    sys.exit(1)

device = torch.cuda.current_device()
props = torch.cuda.get_device_properties(device)
print("\u2714 CUDA available")
print(f"GPU : {props.name}")
print(f"VRAM: {props.total_memory/1024**3:.1f} GB")

for mod in ["bitsandbytes", "peft", "transformers"]:
    try:
        importlib.import_module(mod)
        print(f"\u2714 Imported {mod}")
    except Exception as e:
        print(f"\u2716 Failed to import {mod}: {e}")
PY
ok "Sanity checks complete."
