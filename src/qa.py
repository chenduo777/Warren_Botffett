from __future__ import annotations

from typing import Any, Dict, Optional

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_milvus import Milvus
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIARerank

SYSTEM_PROMPT = (
    "You are Warren Buffett himself, replying in the first person to a reader's question. "
    "Voice: plain language, dry humor, self-deprecating about my mistakes. Refer to "
    "Charlie Munger as 'Charlie' and use 'Charlie and I' naturally. Avoid Wall Street "
    "jargon. "
    "\n\n"
    "STRICT GROUNDING — read carefully:\n"
    "1. Every factual claim (numbers, years, company names, deals, decisions, quotes, "
    "people, anecdotes, metaphors, comparisons) must come from the provided letter "
    "excerpts. The excerpts are your ONLY source of truth.\n"
    "2. Do NOT use outside knowledge about Buffett's biography, prior business "
    "ventures, attributed sayings, or events that are not in the excerpts — even if "
    "you 'know' they are true.\n"
    "3. Do NOT invent metaphors, analogies, or quotes (e.g. 'Mr. Market', 'planting "
    "corn', 'borrowed from Peter Lynch') unless that exact metaphor appears in the "
    "excerpts.\n"
    "4. If a relevant fact, quote, or example is not in the excerpts, omit it. Do "
    "not paraphrase from memory.\n"
    "5. If the excerpts do not contain enough to answer, reply exactly: botffet dont know.\n"
    "6. The excerpts may span many years (1977-2021) of my writing. My views did "
    "evolve — sometimes I changed my mind when better information arrived, sometimes "
    "I refined a position. When excerpts from different years take different stances "
    "on the question, you MUST: (a) use all of them, not just the ones that agree; "
    "(b) walk the reader through what I thought then, what shifted, and what I "
    "concluded later; (c) be explicit about the years tied to each stage of the "
    "thinking. Do not flatten my thinking into a single timeless opinion when the "
    "excerpts show it changed.\n"
    "\n"
    "It is better to give a short, plain answer that stays within the excerpts than "
    "a colorful one that drifts beyond them."
)

DEFAULT_LLM_MODEL = "moonshotai/kimi-k2-instruct"
DEFAULT_RERANK_MODEL = "nvidia/llama-3.2-nv-rerankqa-1b-v2"
RERANK_FETCH_MULTIPLIER = 3  # pull k * 3 candidates before reranking


def _build_filter(
    year: Optional[int] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
) -> Optional[str]:
    """Build a Milvus boolean expression string for the year metadata field.

    Returns None when no constraints are given. Raises if --year is combined
    with --start-year/--end-year.
    """
    if year is not None and (start_year is not None or end_year is not None):
        raise ValueError("--year cannot be combined with --start-year/--end-year")

    if year is not None:
        return f"year == {year}"

    clauses = []
    if start_year is not None:
        clauses.append(f"year >= {start_year}")
    if end_year is not None:
        clauses.append(f"year <= {end_year}")

    if not clauses:
        return None
    return " && ".join(clauses)


def _build_retriever(
    vector_store: Milvus,
    k: int,
    expr: Optional[str],
    use_rerank: bool,
    rerank_model: str,
):
    """Dense retriever, optionally wrapped with NVIDIA cross-encoder rerank.

    With rerank: pull k*RERANK_FETCH_MULTIPLIER candidates, rerank down to k.
    Eval on the Buffett corpus (5-question golden set, 2026-04-25) showed
    rerank lifts Precision@5 from 0.60 to 0.72 and keeps MRR@5 at 1.00 — the
    Chinese-question / English-corpus setup defeats BM25 hybrid, so rerank
    is the production path.
    """
    fetch_k = k * RERANK_FETCH_MULTIPLIER if use_rerank else k
    search_kwargs: Dict[str, Any] = {"k": fetch_k}
    if expr:
        search_kwargs["expr"] = expr
    base = vector_store.as_retriever(search_kwargs=search_kwargs)

    if not use_rerank:
        return base

    reranker = NVIDIARerank(model=rerank_model, top_n=k)
    return ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=base
    )


def answer_question(
    vector_store: Milvus,
    query: str,
    llm_model: str = DEFAULT_LLM_MODEL,
    k: int = 5,
    year: Optional[int] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    use_rerank: bool = True,
    rerank_model: str = DEFAULT_RERANK_MODEL,
) -> dict:
    expr = _build_filter(year=year, start_year=start_year, end_year=end_year)

    retriever = _build_retriever(
        vector_store, k=k, expr=expr,
        use_rerank=use_rerank, rerank_model=rerank_model,
    )
    docs = retriever.invoke(query)
    context = "\n\n".join(
        [f"[year={doc.metadata.get('year')}] {doc.page_content}" for doc in docs]
    )

    rag_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                "問題：{question}\n\n可用參考內容：\n{context}\n\n"
                "請用繁體中文作答，並在最後列出引用年份。",
            ),
        ]
    )
    llm = ChatNVIDIA(
        model=llm_model,
        temperature=0.6,
        top_p=0.9,
        max_tokens=4096,
    )
    rag_chain = rag_prompt | llm | StrOutputParser()
    answer = rag_chain.invoke({"question": query, "context": context or ""})

    years = sorted({doc.metadata.get("year") for doc in docs if doc.metadata.get("year")})

    return {
        "answer": answer,
        "years": years,
        "retrieved_docs": docs,
    }
