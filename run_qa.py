from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.index import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MILVUS_URI,
    collection_exists,
    load_vector_store,
)
from src.qa import DEFAULT_LLM_MODEL, answer_question

ROOT = Path(__file__).resolve().parent


def _configure_api_key() -> None:
    load_dotenv()

    nvidia_api_key = (
        os.getenv("NVIDIA_API_KEY")
        or os.getenv("nvidia_api_key")
        or os.getenv("nvidia_kimi_api_key")
    )

    if not nvidia_api_key:
        raise EnvironmentError(
            "Missing API key. Please set NVIDIA_API_KEY (or nvidia_kimi_api_key) in .env"
        )

    os.environ["NVIDIA_API_KEY"] = nvidia_api_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warren Botffett RAG CLI (query only)")
    parser.add_argument("--query", required=True, help="Question to ask")
    parser.add_argument("--mode", choices=["rag", "agent"], default="rag")
    parser.add_argument("--k", type=int, default=5, help="Top-K chunks to retrieve")
    parser.add_argument("--milvus-uri", default=DEFAULT_MILVUS_URI)
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Must match the model used to build the collection.",
    )
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--year", type=int)
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _configure_api_key()

    if not collection_exists(args.milvus_uri, args.collection_name):
        print(
            f"Collection '{args.collection_name}' not found at {args.milvus_uri}.\n"
            f"Build it first:\n"
            f"  python build_index.py --collection-name {args.collection_name}",
            file=sys.stderr,
        )
        sys.exit(1)

    vector_store = load_vector_store(
        uri=args.milvus_uri,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
    )

    result = answer_question(
        vector_store=vector_store,
        query=args.query,
        llm_model=args.llm_model,
        mode=args.mode,
        k=args.k,
        year=args.year,
        start_year=args.start_year,
        end_year=args.end_year,
    )

    print("\n=== Answer ===")
    print(result["answer"])
    print("\n=== Retrieved Years ===")
    print(", ".join(str(y) for y in result["years"]) if result["years"] else "(none)")


if __name__ == "__main__":
    main()
