# Warren_Botffett-

Minimal LangChain RAG implementation for Buffett letters.

## Agent pipeline

The agent graph runs four nodes: **enrich → rewrite → retrieve → generate**.

### enrich_node (`src/enricher.py`)

Three sub-steps before retrieval:

1. **Ticker extraction** — llama-3.1-70b extracts a ticker symbol (`TICKER:<symbol>` or `NONE`) from the user query.
2. **yfinance fetch** — fetches company fundamentals and formats them with currency labels (non-USD values like TWD are explicitly tagged to prevent scale misreads downstream).
3. **Principle-oriented retrieval query** — llama-4-maverick uses a Chain-of-Thought prompt (`TRANSLATE_PROMPT`) to:
   - Draw on pre-trained world knowledge about the company/industry (e.g. DRAM is cyclical, Costco's low margins are by design)
   - Cross-check against the financial data and flag anomalies (e.g. cycle-peak inflation)
   - Decide retrieval direction: fits Buffett's principles → find praise passages; does not fit → find avoidance passages
   - Output 2–3 sentences of Traditional Chinese prose covering all seven Buffett dimensions: moat, pricing power, ROE, owner earnings, capital allocation, durable competitive advantage, circle of competence

The `[QUERY]` block is extracted by `_parse_query()` with two fallback patterns before falling back to the original user query.

**Key design principle**: The retrieval query targets *Buffett's investment principle passages*, not passages about the queried company. For a commodity business like DRAM, the query finds "why Buffett avoids commodity businesses" — not "semiconductor moat company", which doesn't exist in letters written decades earlier.
