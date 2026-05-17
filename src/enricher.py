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

TRANSLATE_PROMPT = """You are a Warren Buffett investment analysis assistant with deep knowledge of companies and industries.

Task: Generate a Traditional Chinese retrieval query to find the most relevant investment principle passages from Buffett's annual shareholder letters.
Buffett's letters will not mention this company by name; the goal is to retrieve passages where he discusses applicable investment principles.

First reason in the [THINKING] block, then output only the final query after [QUERY].

[THINKING]
1. Draw on your existing knowledge of this company and industry: What is the nature of this business?
   Where does pricing power come from — or why does it not exist? What is the competitive structure?
2. Cross-check against the financial data below. Flag anomalies (e.g., cyclical peak inflating margins).
3. Decide retrieval direction:
   Fits Buffett's principles → find passages where he praises this type of business
   Does not fit → find passages where he explains why he avoids this type of business
[/THINKING]

[QUERY]
(2-3 sentences of connected prose in Traditional Chinese. Must cover all seven dimensions:
strength of moat, presence of pricing power, ROE level, owner earnings characteristics,
capital allocation efficiency, presence of durable competitive advantage, circle of competence.
No questions, no bullet points, no quotation marks.)

Company data:
{company_data}"""

_QUERY_PATTERNS = [
    re.compile(r'\[QUERY\]\s*\n+(.*?)(?=\[|$)', re.DOTALL | re.IGNORECASE),
    re.compile(r'QUERY[:：]\s*\n+(.*?)(?=\[|$)', re.DOTALL | re.IGNORECASE),
]


def _message_text(reply: Any) -> str:
    return reply.content if hasattr(reply, "content") else str(reply)


def _parse_query(raw: str, fallback: str) -> str:
    for pat in _QUERY_PATTERNS:
        m = pat.search(raw)
        if m and m.group(1).strip():
            return m.group(1).strip()
    return fallback


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
    fin_currency = info.get("financialCurrency") or info.get("currency") or "USD"
    price_currency = info.get("currency") or "USD"

    def money(v: Any, currency: str) -> Optional[str]:
        formatted = _format_money(v)
        if formatted is None:
            return None
        if currency != "USD":
            return f"{formatted} {currency}"
        return formatted

    rows = [
        ("Company", info.get("longName")),
        ("Sector / Industry", f"{info.get('sector', 'N/A')} / {info.get('industry', 'N/A')}"),
        ("Business", summary if summary else None),
        ("Revenue", money(info.get("totalRevenue"), fin_currency)),
        ("Gross Margin", _format_percent(info.get("grossMargins"))),
        ("Operating Margin", _format_percent(info.get("operatingMargins"))),
        ("Net Margin", _format_percent(info.get("profitMargins"))),
        ("ROE", _format_percent(info.get("returnOnEquity"))),
        ("Free Cash Flow", money(info.get("freeCashflow"), fin_currency)),
        ("Market Cap", money(info.get("marketCap"), price_currency)),
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
    raw = _message_text(translate_reply)
    retrieval_query = _parse_query(raw, query)
    return retrieval_query, company_data
