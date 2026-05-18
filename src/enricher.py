from __future__ import annotations

import re
from typing import Any, Optional, Tuple

import yfinance as yf
import langchain.agents as langchain_agents
try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
except ImportError:
    from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

    langchain_agents.AgentExecutor = AgentExecutor
    langchain_agents.create_tool_calling_agent = create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

EXTRACT_TICKER_PROMPT = """Extract the public company ticker symbol from the user's query.

Rules:
- If the query clearly asks about one public company, output TICKER:<symbol>.
- Prefer US-listed ticker symbols when obvious.
- If no specific public company is present, output NONE.
- Output only TICKER:<symbol> or NONE.

Query:
{query}"""

ENRICHER_SYSTEM_PROMPT = """You are a financial data collector for a company research pipeline.

Your job is to gather data for ticker {ticker}. Follow these steps exactly once, then stop.

1. Call get_company_snapshot to get current metrics.
2. If the user query mentions long-term holding (5+ years), durability, or multi-year growth, ALSO call get_historical_financials.
3. After calling the tools (at most twice), respond with exactly one word: DONE

Do NOT call any tool more than once. Do NOT write analysis."""

TRANSLATE_PROMPT = """You are a Warren Buffett investment research assistant.

User query: {query}

Company financial data:
{company_data}

Your task: generate a Chroma retrieval query that finds the most relevant passages from Warren Buffett's shareholder letters.

Think step by step:

[THINKING]
1. Business type: what industry/model is this? (commodity, branded consumer, capital-light platform, cyclical semiconductor, etc.)
2. Moat check: does the data show durable pricing power, or is it a cyclical peak? (Revenue CAGR >40% in semiconductors is often a peak-quarter artifact, not a structural moat)
3. Investment horizon: if the user specifies a holding period (e.g. "5–10 years"), assess durability over that horizon — can the moat hold through 2 economic cycles? Is there meaningful disruption risk in that timeframe?
4. Seven Buffett dimensions: moat strength, pricing power, ROE level, owner earnings characteristics, capital allocation efficiency, durable competitive advantage, circle of competence fit.
5. Retrieval direction: does this business FIT Buffett's principles → search for passages where he PRAISES this type of business; or does NOT fit → search for passages where he EXPLAINS why he AVOIDS this type of business.
[/THINKING]

Based on your thinking above, output ONLY this block in Traditional Chinese, 2–3 sentences of connected prose, no bullets, no questions:

[QUERY]
(cover all seven dimensions from step 4; frame toward the retrieval direction from step 5)
[/QUERY]"""

_QUERY_PATTERNS = [
    re.compile(r'\[QUERY\]\s*(.*?)\s*\[/QUERY\]', re.DOTALL | re.IGNORECASE),
    re.compile(r'\[QUERY\]\s*\n+(.*?)(?=\[|$)', re.DOTALL | re.IGNORECASE),
    re.compile(r'QUERY[:：]\s*\n+(.*?)(?=\[|$)', re.DOTALL | re.IGNORECASE),
]


def _message_text(reply: Any) -> str:
    return reply.content if hasattr(reply, 'content') else str(reply)


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
        return f'${amount / 1_000_000_000_000:.2f}T'
    if abs_amount >= 1_000_000_000:
        return f'${amount / 1_000_000_000:.2f}B'
    if abs_amount >= 1_000_000:
        return f'${amount / 1_000_000:.2f}M'
    return f'${amount:,.0f}'


def _format_percent(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f'{number * 100:.1f}%'


def _format_company_data(info: dict[str, Any]) -> str:
    summary = (info.get('longBusinessSummary') or '')[:400]
    fin_currency = info.get('financialCurrency') or info.get('currency') or 'USD'
    price_currency = info.get('currency') or 'USD'

    def money(v: Any, currency: str) -> Optional[str]:
        formatted = _format_money(v)
        if formatted is None:
            return None
        if currency != 'USD':
            return f'{formatted} {currency}'
        return formatted

    rows = [
        ('Company', info.get('longName')),
        ('Sector / Industry', f"{info.get('sector', 'N/A')} / {info.get('industry', 'N/A')}"),
        ('Business', summary if summary else None),
        ('Revenue', money(info.get('totalRevenue'), fin_currency)),
        ('Gross Margin', _format_percent(info.get('grossMargins'))),
        ('Operating Margin', _format_percent(info.get('operatingMargins'))),
        ('Net Margin', _format_percent(info.get('profitMargins'))),
        ('ROE', _format_percent(info.get('returnOnEquity'))),
        ('Free Cash Flow', money(info.get('freeCashflow'), fin_currency)),
        ('Market Cap', money(info.get('marketCap'), price_currency)),
    ]
    return '\n'.join(f'{label}: {value}' for label, value in rows if value)


def _format_historical_financials(ticker_obj: Any) -> str:
    """Return a 4-year annual trend table string, or '' on any error."""
    try:
        fin = ticker_obj.financials
        cf  = ticker_obj.cashflow

        if fin is None or fin.empty:
            return ''

        years = sorted(fin.columns)
        labels = [str(y.year) for y in years]

        def row_vals(df, *keys):
            for k in keys:
                if k in df.index:
                    return [df.loc[k, y] for y in years]
            return None

        revenue = row_vals(fin, 'Total Revenue')
        gp      = row_vals(fin, 'Gross Profit')
        op_inc  = row_vals(fin, 'Operating Income', 'EBIT')
        fcf     = row_vals(cf, 'Free Cash Flow')

        if revenue is None:
            return ''

        def pct(num, denom):
            try:
                return f'{num / denom * 100:.0f}%'
            except Exception:
                return 'N/A'

        def fmt(v):
            return _format_money(v) if v is not None else 'N/A'

        header  = '  '.join(f'FY{l}' for l in labels)
        rev_row = '  '.join(f'{fmt(v):>8}' for v in revenue)
        gm_row  = '  '.join(f'{pct(g, r):>8}' for g, r in zip(gp or [None]*len(revenue), revenue)) if gp else None
        om_row  = '  '.join(f'{pct(o, r):>8}' for o, r in zip(op_inc or [None]*len(revenue), revenue)) if op_inc else None
        fcf_row = '  '.join(f'{fmt(v):>8}' for v in fcf) if fcf else None

        cagr_str = ''
        try:
            r0, rn = float(revenue[0]), float(revenue[-1])
            n = len(revenue) - 1
            if r0 > 0 and rn > 0 and n > 0:
                cagr = (rn / r0) ** (1 / n) - 1
                cagr_str = f'Revenue {n}-yr CAGR: {cagr * 100:.1f}%'
        except Exception:
            pass

        lines = ['--- Annual Trend (oldest → newest) ---', f'           {header}']
        lines.append(f'Revenue    {rev_row}')
        if gm_row:
            lines.append(f'Gross Mg   {gm_row}')
        if om_row:
            lines.append(f'Op Mg      {om_row}')
        if fcf_row:
            lines.append(f'FCF        {fcf_row}')
        if cagr_str:
            lines.append(cagr_str)

        return '\n'.join(lines)
    except Exception:
        return ''


def _parse_ticker(text: str) -> Optional[str]:
    cleaned = text.strip()
    if cleaned.upper() == 'NONE':
        return None

    match = re.search(r'\bTICKER\s*:\s*([A-Z0-9.^=-]+)\b', cleaned, re.IGNORECASE)
    if not match:
        return None
    return match.group(1).upper()


def _build_enricher_tools(ticker: str, tavily_api_key: Optional[str] = None) -> list:
    @tool
    def get_company_snapshot(t: str) -> str:
        """Get current financial snapshot for a public company: revenue, margins, ROE, FCF, market cap."""
        info = yf.Ticker(t).info
        if 'sector' not in info:
            return f'No financial data found for ticker {t}'
        return _format_company_data(info)

    @tool
    def get_historical_financials(t: str) -> str:
        """Get 4-year annual financial trends for a public company: revenue CAGR, margin trajectory, FCF growth."""
        result = _format_historical_financials(yf.Ticker(t))
        return result or f'No historical data available for {t}'

    tools = [get_company_snapshot, get_historical_financials]

    if tavily_api_key:
        @tool
        def search_company_news(query: str) -> str:
            """Search the web for recent news, analyst reports, or competitive analysis about a company."""
            from tavily import TavilyClient
            response = TavilyClient(api_key=tavily_api_key).search(query, max_results=3)
            snippets = [r.get('content', '') for r in response.get('results', [])]
            return '\n\n'.join(snippets) if snippets else 'No results found.'
        tools.append(search_company_news)

    return tools


def enrich_query(
    query: str,
    extractor_llm: Any,
    agent_llm: Any,
    translator_llm: Any = None,
    tavily_api_key: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    extract_reply = extractor_llm.invoke(EXTRACT_TICKER_PROMPT.format(query=query))
    ticker = _parse_ticker(_message_text(extract_reply))
    if ticker is None:
        return query, None

    info = yf.Ticker(ticker).info
    if 'sector' not in info:
        return query, None

    # Phase 1: gather financial data via tool-calling agent (70b)
    tools = _build_enricher_tools(ticker, tavily_api_key)
    prompt = ChatPromptTemplate.from_messages([
        ('system', ENRICHER_SYSTEM_PROMPT.format(ticker=ticker)),
        ('human', '{input}'),
        ('placeholder', '{agent_scratchpad}'),
    ])
    agent = (
        create_tool_calling_agent(agent_llm, tools, prompt)
        if hasattr(agent_llm, 'bind_tools')
        else RunnableLambda(lambda _: {'output': ''})
    )
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        return_intermediate_steps=True,
        max_iterations=3,
    )

    result = executor.invoke({'input': query})

    tool_outputs = [
        str(obs)
        for _, obs in result.get('intermediate_steps', [])
        if obs
    ]
    company_context = '\n\n'.join(tool_outputs) if tool_outputs else None

    # Phase 2: generate Buffett-vocabulary retrieval query
    # Use translator_llm (maverick) if available; fall back to agent's own output.
    if translator_llm and company_context:
        translate_reply = translator_llm.invoke(
            TRANSLATE_PROMPT.format(query=query, company_data=company_context)
        )
        raw = _message_text(translate_reply)
    else:
        raw = result.get('output', '')

    retrieval_query = _parse_query(raw, query)
    return retrieval_query, company_context
