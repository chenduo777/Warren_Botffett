from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from src.agent import build_agent_graph
from src.enricher import enrich_query


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return AIMessage(content=self.replies.pop(0))


class FakeTicker:
    def __init__(self, info):
        self.info = info


def test_enrich_passthrough_when_no_company(monkeypatch):
    extractor = FakeLLM(["NONE"])
    translator = FakeLLM([])

    result = enrich_query("巴菲特怎麼看護城河？", extractor, translator)

    assert result == ("巴菲特怎麼看護城河？", None)
    assert translator.prompts == []


def test_enrich_passthrough_on_invalid_ticker(monkeypatch):
    import src.enricher as enricher

    monkeypatch.setattr(enricher.yf, "Ticker", lambda ticker: FakeTicker({}))
    extractor = FakeLLM(["TICKER:FAKE"])
    translator = FakeLLM([])

    result = enrich_query("FAKE 這家公司如何？", extractor, translator)

    assert result == ("FAKE 這家公司如何？", None)
    assert translator.prompts == []


def test_enrich_returns_translated_query_and_context(monkeypatch):
    import src.enricher as enricher

    info = {
        "longName": "Taiwan Semiconductor Manufacturing Company Limited",
        "sector": "Technology",
        "industry": "Semiconductors",
        "longBusinessSummary": "TSMC manufactures and sells integrated circuits and semiconductor products.",
        "totalRevenue": 2_500_000_000,
        "grossMargins": 0.55,
        "operatingMargins": 0.42,
        "profitMargins": 0.38,
        "returnOnEquity": 0.31,
        "freeCashflow": 800_000_000,
        "marketCap": 600_000_000_000,
    }
    monkeypatch.setattr(enricher.yf, "Ticker", lambda ticker: FakeTicker(info))
    extractor = FakeLLM(["TICKER:TSM"])
    translator = FakeLLM([
        "[THINKING]\nSemiconductor foundry with real moat.\n[/THINKING]\n\n[QUERY]\nTSM semiconductor moat capital allocation"
    ])

    retrieval_query, company_context = enrich_query(
        "台積電的護城河如何？", extractor, translator
    )

    assert retrieval_query == "TSM semiconductor moat capital allocation"
    assert "Company: Taiwan Semiconductor Manufacturing Company Limited" in company_context
    assert "Sector / Industry: Technology / Semiconductors" in company_context
    assert "Business: TSMC manufactures and sells integrated circuits" in company_context
    assert "Revenue: $2.50B" in company_context
    assert "Gross Margin: 55.0%" in company_context
    assert "ROE: 31.0%" in company_context
    assert "Free Cash Flow: $800.00M" in company_context
    assert "Market Cap: $600.00B" in company_context


def test_enrich_fallback_when_query_marker_missing(monkeypatch):
    import src.enricher as enricher

    info = {"sector": "Technology", "industry": "Semiconductors", "longName": "Acme Corp"}
    monkeypatch.setattr(enricher.yf, "Ticker", lambda ticker: FakeTicker(info))
    extractor = FakeLLM(["TICKER:ACME"])
    translator = FakeLLM(["some reply with no query marker at all"])

    original_query = "Acme 這家公司如何？"
    retrieval_query, _ = enrich_query(original_query, extractor, translator)

    assert retrieval_query == original_query


def test_build_agent_graph_returns_compilable_graph(monkeypatch):
    import src.agent as agent
    import src.enricher as enricher

    class FakeChatNVIDIA:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.calls = 0

        def invoke(self, prompt):
            self.calls += 1
            if isinstance(prompt, str):
                return AIMessage(content="NONE")
            return AIMessage(content="answer")

    class FakeRetriever:
        def invoke(self, query):
            return [Document(page_content="Buffett letter excerpt", metadata={"year": 1999})]

    class FakeVectorStore:
        def as_retriever(self, search_kwargs):
            return FakeRetriever()

    monkeypatch.setattr(agent, "ChatNVIDIA", FakeChatNVIDIA)
    monkeypatch.setattr(enricher.yf, "Ticker", lambda ticker: FakeTicker({}))

    graph = build_agent_graph(FakeVectorStore(), use_rerank=False)
    result = graph.invoke(
        {"messages": [HumanMessage(content="巴菲特怎麼看護城河？")]},
        config={"configurable": {"thread_id": "test"}},
    )

    assert result["messages"][-1].content == "answer"
    assert result["retrieved_years"] == [1999]
