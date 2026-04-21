from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from src.ingest import load_letter_documents
from src.index import build_vector_store, load_vector_store, split_documents
from src.qa import answer_question

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_FILE = ROOT / "data" / "letters.txt"
DEFAULT_DB_DIR = ROOT / "chroma_db"


def _configure_api_key() -> None:
    load_dotenv()

    google_api_key = (
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("gemini_api_key")
    )

    if not google_api_key:
        raise EnvironmentError(
            "Missing API key. Please set GOOGLE_API_KEY, GEMINI_API_KEY, or gemini_api_key in .env"
        )

    os.environ["GOOGLE_API_KEY"] = google_api_key


def _build_index_if_needed(
    data_file: Path,
    db_dir: Path,
    embedding_model: str,
    rebuild: bool,
) -> None:
    if db_dir.exists() and not rebuild:
        return

    docs = load_letter_documents(data_file)
    chunks = split_documents(docs)
    build_vector_store(
        chunks=chunks,
        persist_directory=db_dir,
        embedding_model=embedding_model,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warren Botffett RAG CLI")
    parser.add_argument("--query", required=True, help="Question to ask")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _configure_api_key()

    _build_index_if_needed(
        data_file=args.data_file,
        db_dir=args.db_dir,
        embedding_model=args.embedding_model,
        rebuild=args.rebuild,
    )

    vector_store = load_vector_store(
        persist_directory=args.db_dir,
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
