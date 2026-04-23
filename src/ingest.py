from __future__ import annotations

import re
from pathlib import Path
from typing import List

from langchain_core.documents import Document

_DOT_PADDING_RE = re.compile(r"\.{4,}|(?: \. ){3,}")


def looks_like_table_dump(text: str) -> bool:
    """Heuristic: True if `text` looks like financial-table noise rather than narrative.

    A chunk is flagged when ANY of these hold:
      - >30% of tokens are pure numbers (e.g. broken table cells like "5 000 000")
      - has dot-padding ("....." or " . . . . ") used to right-align figures
      - >55% of tokens are 1-2 chars (table fragments, not real prose)

    Short fragments (<5 tokens) are never flagged — too little signal.
    """
    tokens = text.split()
    if len(tokens) < 5:
        return False

    digit_tokens = sum(
        1
        for t in tokens
        if t.replace(",", "").replace(".", "").replace("-", "").isdigit()
    )
    if digit_tokens / len(tokens) > 0.30:
        return True

    if _DOT_PADDING_RE.search(text):
        return True

    short_tokens = sum(1 for t in tokens if len(t) <= 2)
    if short_tokens / len(tokens) > 0.55:
        return True

    return False


def load_letter_documents(file_path: str | Path) -> List[Document]:
    """Load Buffett letters as one line per year with metadata.

    Year mapping rule:
    - line 1 -> 1977
    - year = 1976 + line_number
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"letters file not found: {path}")

    documents: List[Document] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            content = raw_line.strip()
            if not content:
                continue

            year = 1976 + line_number
            documents.append(
                Document(
                    page_content=content,
                    metadata={"year": year, "line_number": line_number},
                )
            )

    if not documents:
        raise ValueError("No non-empty lines found in letters file.")

    return documents

