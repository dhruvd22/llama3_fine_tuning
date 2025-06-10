#!/usr/bin/env python3
"""Download Spider and WikiSQL datasets and merge them for fine-tuning."""

from __future__ import annotations

import argparse
import json
import os
import random
import tarfile
from pathlib import Path
from typing import Iterable

import requests
from tqdm import tqdm


SPIDER_URL = "https://codeload.github.com/taoyds/spider/tar.gz/refs/heads/master"
WIKISQL_URL = "https://codeload.github.com/salesforce/WikiSQL/tar.gz/refs/heads/master"


def download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def extract_spider(src: Path, dest: Path) -> Path:
    target = dest / "spider/train_spider.json"
    with tarfile.open(src, "r:gz") as tar:
        member = tar.getmember(
            "spider-master/evaluation_examples/examples/train_spider.json"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        with tar.extractfile(member) as f, open(target, "wb") as out:
            out.write(f.read())
    return target


def extract_wikisql(src: Path, dest: Path) -> Path:
    target = dest / "wikisql/data/train.jsonl"
    with tarfile.open(src, "r:gz") as tar:
        data_member = tar.getmember("WikiSQL-master/data.tar.bz2")
        with tar.extractfile(data_member) as data_file:
            with tarfile.open(fileobj=data_file, mode="r:bz2") as data_tar:
                member = data_tar.getmember("data/train.jsonl")
                target.parent.mkdir(parents=True, exist_ok=True)
                with data_tar.extractfile(member) as f, open(target, "wb") as out:
                    out.write(f.read())
    return target


def wikisql_sql_to_string(spec: dict) -> str:
    agg_ops = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
    cond_ops = ["=", ">", "<", "OP"]
    sel = f"col{spec['sel']}"
    agg = agg_ops[spec.get("agg", 0)]
    if agg:
        sel = f"{agg}({sel})"
    where = []
    for col, op, val in spec.get("conds", []):
        if isinstance(val, str):
            val = "'" + val.replace("'", "''") + "'"
        where.append(f"col{col} {cond_ops[op]} {val}")
    where_str = ""
    if where:
        where_str = " WHERE " + " AND ".join(where)
    return f"SELECT {sel} FROM table{where_str}"


def load_spider(path: Path) -> Iterable[dict]:
    with open(path) as f:
        data = json.load(f)
    for row in data:
        yield {
            "system": "You translate questions into SQL.",
            "input": row["question"],
            "output": row["query"],
        }


def load_wikisql(path: Path) -> Iterable[dict]:
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            yield {
                "system": "You translate questions into SQL.",
                "input": row["question"],
                "output": wikisql_sql_to_string(row["sql"]),
            }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Spider and WikiSQL")
    parser.add_argument("target_dir", type=Path)
    parser.add_argument("--stage", default="stage1")
    args = parser.parse_args()

    target_dir = args.target_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    spider_tar = target_dir / "spider.tar.gz"
    wikisql_tar = target_dir / "wikisql.tar.gz"

    download(SPIDER_URL, spider_tar)
    download(WIKISQL_URL, wikisql_tar)

    spider_train = extract_spider(spider_tar, target_dir)
    wikisql_train = extract_wikisql(wikisql_tar, target_dir)

    examples = list(load_spider(spider_train))
    examples.extend(load_wikisql(wikisql_train))
    random.shuffle(examples)

    out_file = target_dir / f"merged_{args.stage}.jsonl"
    with open(out_file, "w") as f:
        for ex in tqdm(examples, desc="writing", ncols=80):
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\u2714 Wrote {len(examples)} examples → {out_file}")


if __name__ == "__main__":
    main()
