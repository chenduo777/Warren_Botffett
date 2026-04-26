from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

from src.index import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MILVUS_URI,
    collection_exists,
    load_vector_store,
)
from src.qa import DEFAULT_LLM_MODEL, DEFAULT_RERANK_MODEL, answer_question


def _configure_api_key() -> None:
    load_dotenv()
    key = os.getenv("NVIDIA_API_KEY") or os.getenv("nvidia_api_key")
    if not key:
        raise EnvironmentError("Missing NVIDIA_API_KEY in .env")
    os.environ["NVIDIA_API_KEY"] = key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warren Botffett RAG CLI (query only)")
    parser.add_argument("--query", required=True, help="Question to ask")
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
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip the cross-encoder rerank step (faster, lower precision).",
    )
    parser.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL)
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print the full content of each retrieved chunk before the answer.",
    )
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
        k=args.k,
        year=args.year,
        start_year=args.start_year,
        end_year=args.end_year,
        use_rerank=not args.no_rerank,
        rerank_model=args.rerank_model,
    )

    if args.verbose:
        print("\n=== Retrieved Chunks ===")
        for i, doc in enumerate(result["retrieved_docs"], start=1):
            year = doc.metadata.get("year", "?")
            print(f"\n--- [{i}] year={year} ---")
            print(doc.page_content)

    print("\n=== Answer ===")
    print(result["answer"])
    print("\n=== Retrieved Years ===")
    print(", ".join(str(y) for y in result["years"]) if result["years"] else "(none)")


if __name__ == "__main__":
    main()
