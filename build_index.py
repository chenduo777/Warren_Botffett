from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from src.index import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MILVUS_URI,
    build_vector_store,
    split_documents,
)
from src.ingest import load_letter_documents

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_FILE = ROOT / "data" / "letters.txt"


def _configure_api_key() -> None:
    load_dotenv()
    nvidia_api_key = (
        os.getenv("NVIDIA_API_KEY")
        or os.getenv("nvidia_api_key")
        or os.getenv("nvidia_kimi_api_key")
    )
    if not nvidia_api_key:
        raise EnvironmentError(
            "Missing API key. Set NVIDIA_API_KEY (or nvidia_kimi_api_key) in .env"
        )
    os.environ["NVIDIA_API_KEY"] = nvidia_api_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build (or rebuild) the Milvus collection for Buffett letters."
    )
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--milvus-uri", default=DEFAULT_MILVUS_URI)
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _configure_api_key()

    print(f"Loading letters from {args.data_file} ...")
    docs = load_letter_documents(args.data_file)
    print(f"  -> {len(docs)} letters")

    print("Splitting into chunks ...")
    chunks = split_documents(docs)
    from src.index import _split_documents_last_filtered

    print(
        f"  -> {len(chunks)} chunks kept "
        f"({_split_documents_last_filtered} table-like chunks filtered out)"
    )

    print(f"Embedding and writing to Milvus collection '{args.collection_name}' at {args.milvus_uri} ...")
    print("(any existing collection with this name will be dropped)")
    build_vector_store(
        chunks=chunks,
        uri=args.milvus_uri,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        drop_old=True,
    )
    print("Done.")


if __name__ == "__main__":
    main()
