FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY docs ./docs
COPY tests ./tests

RUN python -m pip install --no-cache-dir -e ".[dev]"

CMD ["python", "-m", "travel_agent.rag.cli", "--help"]
