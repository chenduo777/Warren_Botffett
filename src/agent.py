from __future__ import annotations

from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langchain_milvus import Milvus
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from src.enricher import enrich_query
from src.qa import (
    DEFAULT_RERANK_MODEL,
    HISTORY_TOKEN_BUDGET,
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    ChatState,
    _build_filter,
    _build_retriever,
    make_rewrite_node,
    make_retrieve_node,
)

DEFAULT_AGENT_LLM_MODEL = "meta/llama-4-maverick-17b-128e-instruct"
DEFAULT_HELPER_LLM_MODEL = "meta/llama-3.1-70b-instruct"

ENRICHED_USER_TEMPLATE = (
    "問題：{question}\n\n"
    "【這家公司的當前基本面資料】\n{company_context}\n\n"
    "可用參考內容（來自我歷年股東信）：\n{context}\n\n"
    "請結合以上公司資料與股東信內容，評估這家公司是否符合你的投資原則，"
    "並明確說明你的看法是否因此調整。\n"
    "若股東信中沒有直接討論這家公司，請以第一人稱回答：「雖然我沒有公開買過這家公司，"
    "但根據以上資料和我的投資原則，我認為⋯⋯」。不要回答 botffet dont know。\n"
    "請用繁體中文作答。"
)


class AgentState(ChatState, total=False):
    company_context: Optional[str]


def build_agent_graph(
    vector_store: Milvus,
    llm_model: str = DEFAULT_AGENT_LLM_MODEL,
    k: int = 5,
    year: Optional[int] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    use_rerank: bool = True,
    rerank_model: str = DEFAULT_RERANK_MODEL,
    history_token_budget: int = HISTORY_TOKEN_BUDGET,
    checkpointer: Optional[BaseCheckpointSaver] = None,
):
    expr = _build_filter(year=year, start_year=start_year, end_year=end_year)
    retriever = _build_retriever(
        vector_store,
        k=k,
        expr=expr,
        use_rerank=use_rerank,
        rerank_model=rerank_model,
    )
    llm = ChatNVIDIA(model=llm_model, temperature=0.2, top_p=0.9, max_tokens=4096)
    helper_llm = ChatNVIDIA(
        model=DEFAULT_HELPER_LLM_MODEL,
        temperature=0.0,
        max_completion_tokens=200,
    )

    def enrich_node(state: AgentState) -> dict:
        msgs = state["messages"]
        last_user = next(m for m in reversed(msgs) if isinstance(m, HumanMessage))
        retrieval_query, company_context = enrich_query(
            str(last_user.content),
            helper_llm,   # extractor: llama-3.1-70b (fast, simple)
            llm,           # translator: maverick (knowledge-rich CoT)
        )
        update = {"retrieval_query": retrieval_query}
        if company_context:
            update["company_context"] = company_context
        return update

    _base_rewrite = make_rewrite_node(helper_llm)

    def rewrite_node(state: AgentState) -> dict:
        if state.get("retrieval_query") and state.get("company_context"):
            return {}
        return _base_rewrite(state)

    retrieve_node = make_retrieve_node(retriever)

    def generate_node(state: AgentState) -> dict:
        trimmed = trim_messages(
            state["messages"],
            strategy="last",
            token_counter=count_tokens_approximately,
            max_tokens=history_token_budget,
            start_on="human",
            end_on=("human",),
        )

        docs: List[Document] = state["retrieved_docs"]
        context = "\n\n".join(
            [f"[year={d.metadata.get('year')}] {d.page_content}" for d in docs]
        )

        last_user = trimmed[-1]
        if state.get("company_context"):
            wrapped = HumanMessage(
                content=ENRICHED_USER_TEMPLATE.format(
                    question=last_user.content,
                    company_context=state["company_context"],
                    context=context,
                )
            )
        else:
            wrapped = HumanMessage(
                content=USER_TEMPLATE.format(question=last_user.content, context=context)
            )

        msgs_for_llm = (
            [SystemMessage(content=SYSTEM_PROMPT)]
            + list(trimmed[:-1])
            + [wrapped]
        )

        response = llm.invoke(msgs_for_llm)
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("enrich", enrich_node)
    builder.add_node("rewrite", rewrite_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_edge(START, "enrich")
    builder.add_edge("enrich", "rewrite")
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    return builder.compile(checkpointer=checkpointer or InMemorySaver())
