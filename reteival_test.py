"""Embed a prompt and print the vector for pasting into Attu's Vector Search.

Usage:
    python reteival_test.py "你的問題"
    python reteival_test.py "你的問題" --out vec.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from src.index import DEFAULT_EMBEDDING_MODEL


def _configure_api_key() -> None:
    load_dotenv()
    key = (
        os.getenv("NVIDIA_API_KEY")
        or os.getenv("nvidia_api_key")
        or os.getenv("nvidia_kimi_api_key")
    )
    if not key:
        raise EnvironmentError(
            "Missing API key. Set NVIDIA_API_KEY (or nvidia_api_key) in .env"
        )
    os.environ["NVIDIA_API_KEY"] = key


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Embed a prompt for Attu vector search.")
    p.add_argument("prompt", help="Text to embed")
    p.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional file to write the vector (as JSON array). Otherwise printed to stdout.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _configure_api_key()

    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

    emb = NVIDIAEmbeddings(model=args.embedding_model, truncate="NONE")
    vec = emb.embed_query(args.prompt)

    print(f"prompt   : {args.prompt}")
    print(f"model    : {args.embedding_model}")
    print(f"dim      : {len(vec)}")
    print(f"first 5  : {vec[:5]}")

    payload = json.dumps(vec)
    if args.out:
        args.out.write_text(payload)
        print(f"written  : {args.out}  ({len(payload)} bytes)")
    else:
        print("\n=== full vector (paste into Attu) ===")
        print(payload)


if __name__ == "__main__":
    main()
