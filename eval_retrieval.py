"""LLM-judge evaluation of retrieval quality.

For each question in the golden set, retrieve top-K chunks and have a judge
LLM score each (question, chunk) pair for relevance:

    2 = directly addresses the question with substantive content
    1 = partially relevant — touches the topic but thin or tangential
    0 = not relevant

Metrics (per K):
    Precision@K      : fraction of retrieved chunks scoring >= 1
    HighRelevance@K  : fraction of retrieved chunks scoring == 2
    MeanScore@K      : average score across all retrieved chunks (0-2)
    MRR@K            : reciprocal rank of the first chunk scoring >= 1

Usage:
    python eval_retrieval.py
    python eval_retrieval.py --k 10 --verbose
    python eval_retrieval.py --judge-model moonshotai/kimi-k2-instruct
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Callable, List

from dotenv import load_dotenv

from src.index import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MILVUS_URI,
    load_vector_store,
    split_documents,
)
from src.ingest import load_letter_documents

DEFAULT_DATA_FILE = Path(__file__).resolve().parent / "data" / "letters.txt"

ROOT = Path(__file__).resolve().parent
DEFAULT_GOLDEN = ROOT / "tests" / "eval" / "golden.jsonl"
DEFAULT_JUDGE_MODEL = "moonshotai/kimi-k2-instruct"

JUDGE_PROMPT = """You are evaluating whether a passage from Warren Buffett's shareholder letters is relevant to a user's question.

Question: {question}

What a relevant passage should contain:
{intent}

Passage:
\"\"\"
{passage}
\"\"\"

Score the passage's relevance:
  2 = directly addresses the question with substantive content
  1 = partially relevant — touches the topic but thin or tangential
  0 = not relevant

Respond with ONLY a single digit: 0, 1, or 2. No explanation."""


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


def load_golden(path: Path) -> List[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            items.append(json.loads(line))
    return items


_SCORE_RE = re.compile(r"[012]")


def judge_score(llm, question: str, intent: str, passage: str) -> int:
    prompt = JUDGE_PROMPT.format(
        question=question, intent=intent, passage=passage[:2000]
    )
    backoff = 15.0
    for attempt in range(6):
        try:
            reply = llm.invoke(prompt)
            text = reply.content if hasattr(reply, "content") else str(reply)
            m = _SCORE_RE.search(text)
            return int(m.group(0)) if m else 0
        except Exception as e:
            msg = str(e)
            transient = (
                ("429" in msg) or ("Too Many Requests" in msg)
                or ("502" in msg) or ("503" in msg) or ("504" in msg)
                or ("Bad Gateway" in msg) or ("Service Unavailable" in msg)
                or ("timeout" in msg.lower())
            )
            if transient and attempt < 5:
                print(f"      (rate-limited, sleeping {backoff:.0f}s)")
                time.sleep(backoff)
                backoff = min(backoff * 1.5, 60.0)
                continue
            raise


TRANSLATE_PROMPT = (
    "Translate the following question from Chinese to natural English. "
    "Preserve any English proper nouns as-is. Output ONLY the translation, "
    "no quotes, no explanation.\n\nQuestion: {q}"
)


def make_translator(llm) -> Callable[[str], str]:
    def _translate(q: str) -> str:
        backoff = 15.0
        for attempt in range(6):
            try:
                reply = llm.invoke(TRANSLATE_PROMPT.format(q=q))
                text = reply.content if hasattr(reply, "content") else str(reply)
                return text.strip().strip('"').strip()
            except Exception as e:
                if ("429" in str(e)) and attempt < 5:
                    print(f"      (translate rate-limited, sleeping {backoff:.0f}s)")
                    time.sleep(backoff)
                    backoff = min(backoff * 1.5, 60.0)
                    continue
                raise
    return _translate


def build_retriever(vs, k: int, mode: str, data_file: Path):
    """mode: 'dense' | 'hybrid' | 'translate-hybrid' | 'rerank'"""
    if mode == "dense":
        return vs.as_retriever(search_kwargs={"k": k})

    if mode in ("hybrid", "translate-hybrid"):
        from langchain_community.retrievers import BM25Retriever
        from langchain_classic.retrievers import EnsembleRetriever

        print("Building BM25 index from letters (re-splitting to mirror Milvus chunks)...")
        chunks = split_documents(load_letter_documents(data_file))
        bm25 = BM25Retriever.from_documents(chunks)
        bm25.k = k
        dense = vs.as_retriever(search_kwargs={"k": k})
        return EnsembleRetriever(retrievers=[dense, bm25], weights=[0.5, 0.5])

    if mode == "rerank":
        from langchain_classic.retrievers import ContextualCompressionRetriever
        from langchain_nvidia_ai_endpoints import NVIDIARerank

        # Pull a wider net (k*3) then let the cross-encoder pick the top-k.
        dense_wide = vs.as_retriever(search_kwargs={"k": k * 3})
        reranker = NVIDIARerank(
            model="nvidia/llama-3.2-nv-rerankqa-1b-v2", top_n=k
        )
        return ContextualCompressionRetriever(
            base_compressor=reranker, base_retriever=dense_wide
        )

    return vs.as_retriever(search_kwargs={"k": k})


def evaluate(
    items,
    retriever,
    llm,
    k: int,
    verbose: bool,
    query_transform: Callable[[str], str] = lambda q: q,
) -> None:
    prec_total = 0.0
    high_total = 0.0
    mean_total = 0.0
    mrr_total = 0.0
    n = len(items)

    print(f"\nEvaluating {n} questions @ K={k}  (LLM judge)\n" + "-" * 78)

    for item in items:
        retrieval_query = query_transform(item["question"])
        if verbose and retrieval_query != item["question"]:
            print(f"  [{item['id']:<22}] retrieval_query='{retrieval_query}'")
        docs = retriever.invoke(retrieval_query)[:k]
        scores: List[int] = []
        for d in docs:
            s = judge_score(llm, item["question"], item["intent"], d.page_content)
            scores.append(s)
            time.sleep(2.5)  # NVIDIA NIM free tier is ~ a few RPM

        prec = sum(1 for s in scores if s >= 1) / k
        high = sum(1 for s in scores if s == 2) / k
        mean = sum(scores) / k if scores else 0.0
        mrr = 0.0
        for rank, s in enumerate(scores, start=1):
            if s >= 1:
                mrr = 1.0 / rank
                break

        prec_total += prec
        high_total += high
        mean_total += mean
        mrr_total += mrr

        if verbose:
            years = [d.metadata.get("year") for d in docs]
            print(
                f"  [{item['id']:<22}] scores={scores} "
                f"P={prec:.2f} H={high:.2f} mean={mean:.2f} mrr={mrr:.2f}  "
                f"years={years}"
            )

    print("-" * 78)
    print(f"Precision@{k}     : {prec_total / n:.3f}  (fraction of chunks with score >= 1)")
    print(f"HighRelevance@{k} : {high_total / n:.3f}  (fraction of chunks with score == 2)")
    print(f"MeanScore@{k}     : {mean_total / n:.3f}  (0-2 scale)")
    print(f"MRR@{k}           : {mrr_total / n:.3f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--milvus-uri", default=DEFAULT_MILVUS_URI)
    p.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
    p.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    p.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    p.add_argument(
        "--mode",
        choices=["dense", "hybrid", "translate-hybrid", "rerank"],
        default="dense",
        help="dense=baseline; hybrid=dense+BM25; translate-hybrid=translate Q to EN first; rerank=cross-encoder rerank",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _configure_api_key()

    items = load_golden(args.golden)
    if not items:
        raise SystemExit(f"No items found in {args.golden}")

    vs = load_vector_store(
        uri=args.milvus_uri,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
    )

    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    llm = ChatNVIDIA(model=args.judge_model, temperature=0.0, max_tokens=8)
    retriever = build_retriever(vs, k=args.k, mode=args.mode, data_file=args.data_file)

    query_transform = lambda q: q
    if args.mode == "translate-hybrid":
        translator_llm = ChatNVIDIA(
            model=args.judge_model, temperature=0.0, max_tokens=80
        )
        query_transform = make_translator(translator_llm)

    print(f"Retriever mode: {args.mode}")
    evaluate(
        items, retriever, llm, k=args.k, verbose=args.verbose,
        query_transform=query_transform,
    )


if __name__ == "__main__":
    main()
