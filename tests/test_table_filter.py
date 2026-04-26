"""Tests for the table-dump heuristic used to filter ingest noise."""
from src.ingest import looks_like_table_dump


# ---- should be flagged as TABLE -----------------------------------------


def test_flags_typical_holdings_table_row():
    text = (
        "5.6 2 133 1 884 200 000 000 The Coca Cola Company "
        "8.4 1 299 8 062 6 708 760 M T Bank Corporation "
        "6.0 103 732 48 000 000 Moody's Corporation"
    )
    assert looks_like_table_dump(text)


def test_flags_dot_padded_alignment():
    text = "The Coca Cola Company .......................... 8.4 1299 8062"
    assert looks_like_table_dump(text)


def test_flags_long_run_of_split_numbers():
    text = "1 884 200 000 000 1 299 8 062 6 708 760 103 732 48 000 000"
    assert looks_like_table_dump(text)


# ---- should NOT be flagged ----------------------------------------------


def test_keeps_pure_narrative():
    text = (
        "It's hard to overemphasize the importance of who is CEO of a company. "
        "Before Jim Kilts arrived at Gillette in 2001 the company was struggling, "
        "having particularly suffered from capital allocation blunders."
    )
    assert not looks_like_table_dump(text)


def test_keeps_narrative_with_a_few_years():
    text = (
        "In 1985 we bought our first shares; by 1990 the position had doubled. "
        "Charlie and I believed then, as we believe now, that great businesses "
        "compound value over decades."
    )
    assert not looks_like_table_dump(text)


def test_short_fragments_not_flagged():
    # Too few tokens to judge — must err on the side of keeping.
    assert not looks_like_table_dump("8.4 1299")
    assert not looks_like_table_dump("")


def test_keeps_text_with_one_inline_figure():
    text = (
        "Berkshire reported earnings of 5.0 billion dollars after the merger, "
        "but Charlie and I want to emphasize that this bookkeeping entry "
        "is meaningless from an economic standpoint."
    )
    assert not looks_like_table_dump(text)


def test_flags_mixed_table_head_with_narrative_tail():
    """Holdings tables often end with a narrative paragraph; the chunk
    must still be flagged. Reproduces the 2009/2013/2017 tech-stocks
    failure where mixed chunks displaced real narrative from top-K."""
    table_head = (
        "Johnson Johnson . . 1.0 1 724 1 838 130 272 500 "
        "Kraft Foods Inc. . . 8.8 4 330 3 541 3 947 554 "
        "POSCO . . 5.2 768 2 092 83 128 411 "
        "The Procter Gamble Company . . 2.9 533 5 040 25 108 967 "
        "Wal Mart Stores Inc. . . 1.0 1 893 2 087 334 235 585"
    )
    narrative_tail = (
        " Charlie and I view the marketable common stocks that Berkshire "
        "owns as interests in businesses, not as ticker symbols to be "
        "bought or sold based on chart patterns or analyst targets."
    )
    assert looks_like_table_dump(table_head + narrative_tail)
