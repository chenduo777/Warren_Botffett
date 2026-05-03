from __future__ import annotations

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from langgraph.checkpoint.postgres import PostgresSaver

from src.bot.auth import load_allowed_user_ids
from src.bot.telegram_bot import run_bot
from src.index import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MILVUS_URI,
    collection_exists,
    load_vector_store,
)
from src.qa import DEFAULT_LLM_MODEL, DEFAULT_RERANK_MODEL, build_chat_graph


def _configure_api_key() -> None:
    key = os.getenv("NVIDIA_API_KEY") or os.getenv("nvidia_api_key")
    if not key:
        raise EnvironmentError("Missing NVIDIA_API_KEY in .env")
    os.environ["NVIDIA_API_KEY"] = key


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # python-telegram-bot's HTTP layer is chatty at INFO; quiet it down.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("telegram.ext.Application").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Warren Botffett — Telegram bot (long polling, runs on this machine). "
            "Requires TELEGRAM_BOT_TOKEN and ALLOWED_TG_USER_IDS in .env."
        )
    )
    parser.add_argument("--k", type=int, default=5, help="Top-K chunks to retrieve")
    parser.add_argument("--milvus-uri", default=DEFAULT_MILVUS_URI)
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip the cross-encoder rerank step (faster, lower precision).",
    )
    parser.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL)
    return parser.parse_args()


def main() -> None:
    _configure_logging()
    args = parse_args()
    _configure_api_key()

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise EnvironmentError("Missing TELEGRAM_BOT_TOKEN in .env")

    allowed = load_allowed_user_ids()
    if not allowed:
        raise EnvironmentError(
            "ALLOWED_TG_USER_IDS is empty in .env. Set it to a comma-separated "
            "list of Telegram user_ids (at minimum your own)."
        )

    if not collection_exists(args.milvus_uri, args.collection_name):
        print(
            f"Collection '{args.collection_name}' not found at {args.milvus_uri}.\n"
            f"Build it first:\n"
            f"  python build_index.py --collection-name {args.collection_name}",
            file=sys.stderr,
        )
        sys.exit(1)

    postgres_url = os.getenv("POSTGRES_URL")
    if not postgres_url:
        raise EnvironmentError(
            "Missing POSTGRES_URL in .env. Example:\n"
            "  POSTGRES_URL=postgresql://bot:bot@127.0.0.1:5432/bot?sslmode=disable"
        )

    vector_store = load_vector_store(
        uri=args.milvus_uri,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
    )

    # PostgresSaver owns a connection pool; keep it alive for the whole bot
    # lifetime by wrapping run_bot inside the context manager.
    with PostgresSaver.from_conn_string(postgres_url) as checkpointer:
        checkpointer.setup()  # idempotent; creates checkpoint tables on first run

        graph = build_chat_graph(
            vector_store=vector_store,
            llm_model=args.llm_model,
            k=args.k,
            use_rerank=not args.no_rerank,
            rerank_model=args.rerank_model,
            checkpointer=checkpointer,
        )

        run_bot(graph, token=token, allowed_user_ids=allowed)


if __name__ == "__main__":
    main()
