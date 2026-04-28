from __future__ import annotations

import os
from typing import Optional


def parse_allowed_user_ids(raw: Optional[str]) -> set[int]:
    """Parse a comma-separated list of Telegram user_ids into a set of ints.

    Empty or whitespace-only entries are skipped. A non-numeric entry raises
    ValueError so a typo in `.env` fails loudly instead of silently letting
    nobody (or everybody) in.
    """
    if not raw:
        return set()
    return {int(part.strip()) for part in raw.split(",") if part.strip()}


def load_allowed_user_ids() -> set[int]:
    return parse_allowed_user_ids(os.getenv("ALLOWED_TG_USER_IDS"))
