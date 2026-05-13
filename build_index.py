from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.env import configure_api_key
from src.index import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MILVUS_URI,
    build_vector_store,
    collection_exists,
    split_documents,
)
from src.ingest import load_letter_documents

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_FILE = ROOT / "data" / "letters.txt"




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build (or rebuild) the Milvus collection for Buffett letters."
    )
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--milvus-uri", default=DEFAULT_MILVUS_URI)
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Drop and rebuild the collection even if it already exists. "
             "Without this flag, the script exits 0 when the collection is present "
             "so it is safe to wire as a one-shot init step in docker-compose.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Idempotent fast-path: skip embedding work entirely if the collection
    # is already there. This lets compose run us as an `index-init` service
    # on every `up` without re-burning embedding API credits.
    if not args.force and collection_exists(args.milvus_uri, args.collection_name):
        print(
            f"Collection '{args.collection_name}' already exists at {args.milvus_uri}. "
            f"Skipping. Pass --force to rebuild from scratch."
        )
        return

    configure_api_key()

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
