from __future__ import annotations

from typing import Any, Dict, Optional

from langchain.agents import create_agent
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

SYSTEM_PROMPT = (
    "You are an expert familiar with Warren Buffett's investment philosophy. "
    "Please answer based solely on the provided content of the shareholder letters. "
    "If the content does not contain the answer, please reply directly: botffet dont know."
)


def _build_filter(
    year: Optional[int] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    if year is not None and (start_year is not None or end_year is not None):
        raise ValueError("--year cannot be combined with --start-year/--end-year")

    if year is not None:
        return {"year": year}

    if start_year is not None and end_year is not None:
        return {"$and": [{"year": {"$gte": start_year}}, {"year": {"$lte": end_year}}]}
    if start_year is not None:
        return {"year": {"$gte": start_year}}
    if end_year is not None:
        return {"year": {"$lte": end_year}}

    return None


def answer_question(
    vector_store: Chroma,
    query: str,
    llm_model: str = "gemini-1.5-pro",
    mode: str = "rag",
    k: int = 5,
    year: Optional[int] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
) -> dict:
    where_filter = _build_filter(year=year, start_year=start_year, end_year=end_year)

    search_kwargs: Dict[str, Any] = {"k": k}
    if where_filter:
        search_kwargs["filter"] = where_filter

    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
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
    llm = ChatGoogleGenerativeAI(model=llm_model, temperature=0.1)
    rag_chain = rag_prompt | llm | StrOutputParser()
    answer = rag_chain.invoke({"question": query, "context": context or ""})

    years = sorted({doc.metadata.get("year") for doc in docs if doc.metadata.get("year")})

    return {
        "answer": answer,
        "years": years,
        "retrieved_docs": docs,
    }
