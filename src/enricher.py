from __future__ import annotations

import re
from typing import Any, Optional, Tuple

import yfinance as yf

EXTRACT_TICKER_PROMPT = """Extract the public company ticker symbol from the user's query.

Rules:
- If the query clearly asks about one public company, output TICKER:<symbol>.
- Prefer US-listed ticker symbols when obvious.
- If no specific public company is present, output NONE.
- Output only TICKER:<symbol> or NONE.

Query:
{query}"""

TRANSLATE_PROMPT = """你是巴菲特投資分析助理。根據以下公司資料，
用巴菲特的分析語彙（護城河、定價能力、ROE、資本配置、
耐久型競爭優勢、owner earnings、圈子能力）寫出 2-3 句描述，
作為搜尋巴菲特股東信的關鍵詞句。只輸出描述本身，不要解釋、不要問句。

{company_data}"""


def _message_text(reply: Any) -> str:
    return reply.content if hasattr(reply, "content") else str(reply)


def _format_money(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return str(value)

    abs_amount = abs(amount)
    if abs_amount >= 1_000_000_000_000:
        return f"${amount / 1_000_000_000_000:.2f}T"
    if abs_amount >= 1_000_000_000:
        return f"${amount / 1_000_000_000:.2f}B"
    if abs_amount >= 1_000_000:
        return f"${amount / 1_000_000:.2f}M"
    return f"${amount:,.0f}"


def _format_percent(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number * 100:.1f}%"


def _format_company_data(info: dict[str, Any]) -> str:
    summary = (info.get("longBusinessSummary") or "")[:400]
    rows = [
        ("Company", info.get("longName")),
        ("Sector / Industry", f"{info.get('sector', 'N/A')} / {info.get('industry', 'N/A')}"),
        ("Business", summary if summary else None),
        ("Revenue", _format_money(info.get("totalRevenue"))),
        ("Gross Margin", _format_percent(info.get("grossMargins"))),
        ("Operating Margin", _format_percent(info.get("operatingMargins"))),
        ("Net Margin", _format_percent(info.get("profitMargins"))),
        ("ROE", _format_percent(info.get("returnOnEquity"))),
        ("Free Cash Flow", _format_money(info.get("freeCashflow"))),
        ("Market Cap", _format_money(info.get("marketCap"))),
    ]
    return "\n".join(f"{label}: {value}" for label, value in rows if value)


def _parse_ticker(text: str) -> Optional[str]:
    cleaned = text.strip()
    if cleaned.upper() == "NONE":
        return None

    match = re.search(r"\bTICKER\s*:\s*([A-Z0-9.^=-]+)\b", cleaned, re.IGNORECASE)
    if not match:
        return None
    return match.group(1).upper()


def enrich_query(query: str, extractor_llm: Any, translator_llm: Any) -> Tuple[str, Optional[str]]:
    extract_reply = extractor_llm.invoke(EXTRACT_TICKER_PROMPT.format(query=query))
    ticker = _parse_ticker(_message_text(extract_reply))
    if ticker is None:
        return query, None

    info = yf.Ticker(ticker).info
    if "sector" not in info:
        return query, None

    company_data = _format_company_data(info)
    translate_reply = translator_llm.invoke(
        TRANSLATE_PROMPT.format(company_data=company_data)
    )
    retrieval_query = _message_text(translate_reply).strip()
    return retrieval_query or query, company_data
