from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.documents import Document
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langchain_milvus import Milvus
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIARerank
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages

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
    "7. NUMBERS — handle with extra care:\n"
    "   (a) Every number you state (percentage, dollar amount, share count, date) "
    "must be quoted verbatim from a single excerpt, together with that excerpt's "
    "year. Do not paraphrase numbers from memory.\n"
    "   (b) Do NOT compute deltas, growth rates, or direction-of-change phrases "
    "('rose to', 'fell to', 'grew by') across excerpts. Only state a change if a "
    "single excerpt explicitly states it (e.g. an excerpt that says 'up from "
    "5.39 a year earlier').\n"
    "   (c) Do NOT mix similar-looking numbers across excerpts (e.g. 775 vs 785, "
    "5.4 vs 5.55). If you are unsure which year a number belongs to, omit it.\n"
    "   (d) If two excerpts give different values for the same quantity (e.g. "
    "ownership percentage in different years), present each verbatim with its own "
    "year — do not pick one, average them, or guess the trajectory.\n"
    "   (e) Distinguish 'annual average' / 'cumulative' / 'this year' wording in "
    "the excerpt — never collapse them into a single-year figure.\n"
    "\n"
    "It is better to give a short, plain answer that stays within the excerpts than "
    "a colorful one that drifts beyond them."
)

DEFAULT_LLM_MODEL = "meta/llama-4-maverick-17b-128e-instruct"  # 備用 moonshotai/kimi-k2-instruct
DEFAULT_RERANK_MODEL = "nvidia/llama-3.2-nv-rerankqa-1b-v2"
RERANK_FETCH_MULTIPLIER = 3  # pull k * 3 candidates before reranking
HISTORY_TOKEN_BUDGET = 4000  # trim chat history to this many tokens before LLM call
REWRITE_HISTORY_TURNS = 4    # rewrite_node only reads the last N turns (= 2N messages).
                             # Older turns get dropped before being shown to the rewriter
                             # so persistent (Postgres) history doesn't leak stale topics
                             # into the standalone search query.

USER_TEMPLATE = (
    "問題：{question}\n\n可用參考內容：\n{context}\n\n"
    "請用繁體中文作答。"
)

REWRITE_PROMPT = """You rewrite a user's latest chatbot message into a standalone search query for a Warren Buffett shareholder-letter retrieval system.

Rules:
- Resolve pronouns and back-references ("他", "她", "這家公司", "那個", "剛剛", "前面那筆") to the entities they actually refer to in the chat history.
- Carry forward any topic / company / year from earlier turns that the latest message implicitly depends on.
- Keep it one short sentence in the SAME language as the latest message.
- If the latest message is already fully self-contained, output it unchanged.

Output ONLY the rewritten query. No preamble, no quotes, no explanation.

--- Chat history ---
{history}

--- Latest user message ---
{latest}"""


class ChatState(TypedDict):
    """State for the chat graph.

    `messages` accumulates across turns via `add_messages` reducer.
    `retrieved_docs` / `retrieved_years` / `retrieval_query` are replaced
    fresh each turn — they describe the latest turn only.
    """
    messages: Annotated[list[AnyMessage], add_messages]
    retrieval_query: str
    retrieved_docs: List[Document]
    retrieved_years: List[int]


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


def build_chat_graph(
    vector_store: Milvus,
    llm_model: str = DEFAULT_LLM_MODEL,
    k: int = 5,
    year: Optional[int] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    use_rerank: bool = True,
    rerank_model: str = DEFAULT_RERANK_MODEL,
    history_token_budget: int = HISTORY_TOKEN_BUDGET,
    checkpointer: Optional[BaseCheckpointSaver] = None,
):
    """Build a LangGraph state graph: rewrite -> retrieve -> generate.

    Each session uses a unique `thread_id` (passed in invoke config). Messages
    accumulate per-thread; older history above `history_token_budget` is
    trimmed before each LLM call to bound context size.

    The `rewrite` node turns follow-up messages with pronouns ("剛剛那家公司",
    "他什麼時候買的") into standalone retrieval queries by reading prior turns.
    On the first turn it short-circuits and skips the LLM call.

    If `checkpointer` is None, an in-process `InMemorySaver` is used (history
    is lost on restart). Pass a `PostgresSaver` (already entered as a context
    manager) to persist threads across restarts.
    """
    expr = _build_filter(year=year, start_year=start_year, end_year=end_year)
    retriever = _build_retriever(
        vector_store, k=k, expr=expr,
        use_rerank=use_rerank, rerank_model=rerank_model,
    )
    llm = ChatNVIDIA(model=llm_model, temperature=0.2, top_p=0.9, max_tokens=4096)
    rewriter_llm = ChatNVIDIA(model=llm_model, temperature=0.0, max_tokens=120)

    def rewrite_node(state: ChatState) -> dict:
        msgs = state["messages"]
        last_user_idx = max(
            i for i, m in enumerate(msgs) if isinstance(m, HumanMessage)
        )
        last_user = msgs[last_user_idx]

        # First turn — no prior context, skip the LLM call.
        prior_human = sum(
            1 for m in msgs[:last_user_idx] if isinstance(m, HumanMessage)
        )
        if prior_human == 0:
            return {"retrieval_query": last_user.content}

        # Build a compact history string. Cap to the most recent N turns so
        # long persisted threads don't drown the rewriter in stale topics,
        # and truncate assistant turns to keep token usage low.
        recent = msgs[:last_user_idx][-(REWRITE_HISTORY_TURNS * 2):]
        lines = []
        for m in recent:
            role = "User" if isinstance(m, HumanMessage) else "Assistant"
            content = m.content if isinstance(m, HumanMessage) else m.content[:400]
            lines.append(f"{role}: {content}")
        history_text = "\n".join(lines)

        reply = rewriter_llm.invoke(
            REWRITE_PROMPT.format(history=history_text, latest=last_user.content)
        )
        text = reply.content if hasattr(reply, "content") else str(reply)
        rewritten = text.strip().strip('"').strip("'").strip()
        # Defensive: empty or absurdly long rewrites fall back to the original.
        if not rewritten or len(rewritten) > 300:
            rewritten = last_user.content
        return {"retrieval_query": rewritten}

    def retrieve_node(state: ChatState) -> dict:
        query = state["retrieval_query"]
        docs = retriever.invoke(query)
        years = sorted({d.metadata.get("year") for d in docs if d.metadata.get("year")})
        return {"retrieved_docs": list(docs), "retrieved_years": years}

    def generate_node(state: ChatState) -> dict:
        trimmed = trim_messages(
            state["messages"],
            strategy="last",
            token_counter=count_tokens_approximately,
            max_tokens=history_token_budget,
            start_on="human",
            end_on=("human",),
        )

        docs = state["retrieved_docs"]
        context = "\n\n".join(
            [f"[year={d.metadata.get('year')}] {d.page_content}" for d in docs]
        )

        # Wrap the latest user question with the retrieved context block;
        # earlier turns stay as plain history. The wrapped form is only used
        # for this single LLM call — only the AI response is persisted.
        last_user = trimmed[-1]
        wrapped = HumanMessage(content=USER_TEMPLATE.format(
            question=last_user.content, context=context
        ))
        msgs_for_llm = (
            [SystemMessage(content=SYSTEM_PROMPT)]
            + list(trimmed[:-1])
            + [wrapped]
        )

        response = llm.invoke(msgs_for_llm)
        return {"messages": [response]}

    builder = StateGraph(ChatState)
    builder.add_node("rewrite", rewrite_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_edge(START, "rewrite")
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("retrieve", "generate")
    return builder.compile(checkpointer=checkpointer or InMemorySaver())
