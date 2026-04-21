from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document


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

