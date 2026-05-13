from __future__ import annotations

import os


def configure_api_key() -> None:
    """Resolve NVIDIA_API_KEY from any known .env spelling and normalise to uppercase.

    Checks NVIDIA_API_KEY, nvidia_api_key, and nvidia_kimi_api_key in that order.
    Callers are responsible for calling load_dotenv() before this function.
    """
    key = (
        os.getenv("NVIDIA_API_KEY")
        or os.getenv("nvidia_api_key")
        or os.getenv("nvidia_kimi_api_key")
    )
    if not key:
        raise EnvironmentError(
            "Missing API key. Set NVIDIA_API_KEY in .env"
        )
    os.environ["NVIDIA_API_KEY"] = key
