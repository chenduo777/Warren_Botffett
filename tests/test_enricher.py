import pandas as pd
import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from src.agent import build_agent_graph
from src.enricher import _format_historical_financials, enrich_query


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return AIMessage(content=self.replies.pop(0))


class FakeTicker:
    def __init__(self, info, financials=None, cashflow=None):
        self.info = info
        self.financials = financials if financials is not None else pd.DataFrame()
        self.cashflow   = cashflow   if cashflow   is not None else pd.DataFrame()


def test_enrich_passthrough_when_no_company(monkeypatch):
    extractor = FakeLLM(['NONE'])
    translator = FakeLLM([])

    result = enrich_query('巴菲特怎麼看護城河？', extractor, translator)

    assert result == ('巴菲特怎麼看護城河？', None)
    assert translator.prompts == []


def test_enrich_passthrough_on_invalid_ticker(monkeypatch):
    import src.enricher as enricher

    monkeypatch.setattr(enricher.yf, 'Ticker', lambda ticker: FakeTicker({}))
    extractor = FakeLLM(['TICKER:FAKE'])
    translator = FakeLLM([])

    result = enrich_query('FAKE 這家公司如何？', extractor, translator)

    assert result == ('FAKE 這家公司如何？', None)
    assert translator.prompts == []


def test_enrich_returns_translated_query_and_context(monkeypatch):
    import src.enricher as enricher
    from langchain.agents import AgentExecutor

    info = {
        'longName': 'NVIDIA Corporation',
        'sector': 'Technology',
        'industry': 'Semiconductors',
        'longBusinessSummary': 'NVIDIA designs GPUs and system-on-chip units.',
        'totalRevenue': 130_500_000_000,
        'grossMargins': 0.75,
        'operatingMargins': 0.62,
        'profitMargins': 0.55,
        'returnOnEquity': 1.23,
        'freeCashflow': 60_200_000_000,
        'marketCap': 3_300_000_000_000,
    }
    monkeypatch.setattr(enricher.yf, 'Ticker', lambda t: FakeTicker(info))

    extractor = FakeLLM(['TICKER:NVDA'])
    translator = FakeLLM(['[QUERY]\nNVIDIA 半導體護城河與週期性評估\n[/QUERY]'])

    def fake_invoke(self, inputs):
        return {
            'output': 'agent raw output (not used when translator_llm is provided)',
            'intermediate_steps': [
                (None, 'Company: NVIDIA Corporation\nRevenue: $130.50B'),
                (None, '--- Annual Trend (oldest → newest) ---\nRevenue 3-yr CAGR: 48.8%'),
            ],
        }

    monkeypatch.setattr(AgentExecutor, 'invoke', fake_invoke)

    retrieval_query, company_context = enrich_query(
        'NVDA 適合長期投資嗎？', extractor, FakeLLM([]), translator_llm=translator, tavily_api_key=None
    )

    assert retrieval_query == 'NVIDIA 半導體護城河與週期性評估'
    assert 'NVIDIA' in company_context
    assert 'Annual Trend' in company_context
    assert len(translator.prompts) == 1  # translator was called for Phase 2


def test_enrich_fallback_when_query_marker_missing(monkeypatch):
    import src.enricher as enricher
    from langchain.agents import AgentExecutor

    info = {'sector': 'Technology', 'industry': 'Semiconductors', 'longName': 'Acme Corp'}
    monkeypatch.setattr(enricher.yf, 'Ticker', lambda t: FakeTicker(info))

    def fake_invoke(self, inputs):
        return {
            'output': 'some reply with no query marker at all',
            'intermediate_steps': [],
        }

    monkeypatch.setattr(AgentExecutor, 'invoke', fake_invoke)

    original_query = 'Acme 這家公司如何？'
    retrieval_query, _ = enrich_query(original_query, FakeLLM(['TICKER:ACME']), FakeLLM([]))

    assert retrieval_query == original_query


def test_format_historical_financials_empty():
    class T:
        financials = pd.DataFrame()
        cashflow   = pd.DataFrame()

    assert _format_historical_financials(T()) == ''


def test_format_historical_financials_with_data():
    dates = pd.to_datetime(['2022-01-31', '2023-01-31', '2024-01-31', '2025-01-31'])
    fin = pd.DataFrame(
        {
            dates[0]: [26_900_000_000, 17_475_000_000, 7_000_000_000],
            dates[1]: [44_900_000_000, 25_561_000_000, 7_200_000_000],
            dates[2]: [60_900_000_000, 44_273_000_000, 32_900_000_000],
            dates[3]: [130_500_000_000, 97_875_000_000, 81_000_000_000],
        },
        index=['Total Revenue', 'Gross Profit', 'Operating Income'],
    )
    cf = pd.DataFrame(
        {
            dates[0]: [3_800_000_000],
            dates[1]: [3_800_000_000],
            dates[2]: [27_000_000_000],
            dates[3]: [60_200_000_000],
        },
        index=['Free Cash Flow'],
    )

    class T:
        financials = fin
        cashflow   = cf

    result = _format_historical_financials(T())
    assert 'Annual Trend' in result
    assert 'CAGR' in result
    assert 'FY2022' in result
    assert 'FY2025' in result


def test_build_agent_graph_returns_compilable_graph(monkeypatch):
    import src.agent as agent
    import src.enricher as enricher
    from langchain.agents import AgentExecutor

    class FakeChatNVIDIA:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.calls = 0

        def invoke(self, prompt):
            self.calls += 1
            if isinstance(prompt, str):
                return AIMessage(content='NONE')
            return AIMessage(content='answer')

        def bind_tools(self, tools):
            return self

    class FakeRetriever:
        def invoke(self, query):
            return [Document(page_content='Buffett letter excerpt', metadata={'year': 1999})]

    class FakeVectorStore:
        def as_retriever(self, search_kwargs):
            return FakeRetriever()

    monkeypatch.setattr(agent, 'ChatNVIDIA', FakeChatNVIDIA)
    monkeypatch.setattr(enricher.yf, 'Ticker', lambda ticker: FakeTicker({}))

    def fake_invoke(self, inputs):
        return {'output': 'NONE', 'intermediate_steps': []}

    monkeypatch.setattr(AgentExecutor, 'invoke', fake_invoke)

    graph = build_agent_graph(FakeVectorStore(), use_rerank=False)
    result = graph.invoke(
        {'messages': [HumanMessage(content='巴菲特怎麼看護城河？')]},
        config={'configurable': {'thread_id': 'test'}},
    )

    assert result['messages'][-1].content == 'answer'
    assert result['retrieved_years'] == [1999]
