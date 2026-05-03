FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Non-root runtime user (uid:gid 1000:1000)
RUN groupadd --system --gid 1000 app \
 && useradd  --system --uid 1000 --gid app --create-home --home-dir /home/app app

WORKDIR /app

# Install dependencies first so the layer caches across source-only edits
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Project source. data/ is needed for build_index.py only; bot reads from Milvus.
COPY src/ ./src/
COPY data/ ./data/
COPY run_bot.py build_index.py ./

USER app

# Long-polling Telegram bot — no inbound port. Override CMD to run build_index.py:
#   docker compose run --rm bot python build_index.py
CMD ["python", "run_bot.py"]
