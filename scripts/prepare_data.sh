#!/usr/bin/env bash
set -euo pipefail

ok() { echo -e "\e[32m✔\e[0m $*"; }
err() { echo -e "\e[31m✖\e[0m $*"; }
info() { echo -e "\e[34m➜\e[0m $*"; }

usage() {
  echo "Usage: $0 TARGET_DIR [MODEL_STAGE]" >&2
  exit 1
}

[[ $# -ge 1 ]] || usage
TARGET_DIR="$1"
MODEL_STAGE="${2:-stage1}"

SPIDER_URL="https://yale-lily.github.io/spider/spider.zip"
WIKISQL_URL="https://storage.googleapis.com/wikisql/wikisql.tar.bz2"

mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

info "Checking Spider download..."
if [[ -f spider.zip ]]; then
  ok "spider.zip already exists"
else
  info "🔄 Downloading Spider..."
  wget -q "$SPIDER_URL"
  ok "Spider downloaded"
fi

info "Checking WikiSQL download..."
if [[ -f wikisql.tar.bz2 ]]; then
  ok "wikisql.tar.bz2 already exists"
else
  info "🔄 Downloading WikiSQL..."
  wget -q "$WIKISQL_URL"
  ok "WikiSQL downloaded"
fi

info "Checking Spider extraction..."
if [[ -d spider ]]; then
  ok "Spider already extracted"
else
  info "🔄 Extracting Spider..."
  unzip -q spider.zip
  ok "Spider extracted"
fi

info "Checking WikiSQL extraction..."
if [[ -d wikisql ]]; then
  ok "WikiSQL already extracted"
else
  info "🔄 Extracting WikiSQL..."
  tar -xf wikisql.tar.bz2
  ok "WikiSQL extracted"
fi

info "Building merged JSONL..."
TARGET_DIR="$TARGET_DIR" MODEL_STAGE="$MODEL_STAGE" python3 - <<'PY'
import json, random, os, sys

target_dir = os.environ['TARGET_DIR']
stage = os.environ['MODEL_STAGE']

records = []

spider_path = os.path.join('spider', 'train_spider.json')
with open(spider_path) as f:
    for item in json.load(f):
        records.append({
            'system': 'You translate questions into SQL.',
            'input': item['question'],
            'output': item['query'],
        })

wikisql_path = None
for root, _, files in os.walk('wikisql'):
    if 'train.jsonl' in files:
        wikisql_path = os.path.join(root, 'train.jsonl')
        break
if not wikisql_path:
    print('train.jsonl not found in WikiSQL', file=sys.stderr)
    sys.exit(1)

with open(wikisql_path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        query = item.get('query')
        if isinstance(query, dict):
            query = query.get('human_readable') or query.get('query')
        if query is None and isinstance(item.get('sql'), str):
            query = item['sql']
        elif query is None and isinstance(item.get('sql'), dict):
            query = item['sql'].get('human_readable') or item['sql'].get('query')
        records.append({
            'system': 'You translate questions into SQL.',
            'input': item['question'],
            'output': query,
        })

random.shuffle(records)

out_path = os.path.join(target_dir, f"merged_{stage}.jsonl")
with open(out_path, 'w') as out:
    for rec in records:
        out.write(json.dumps(rec, ensure_ascii=False) + '\n')

print(f"{len(records)} examples written to {out_path}")
PY
ok "JSONL written"

echo "Done! You can now point Axolotl to $TARGET_DIR/merged_${MODEL_STAGE}.jsonl"
