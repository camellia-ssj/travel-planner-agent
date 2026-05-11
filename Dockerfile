# =============================================================================
# travel-agent Dockerfile (multi-stage build)
# =============================================================================
# 使用方式:
#   docker build -t travel-agent .
#   docker run -e DASHSCOPE_API_KEY=xxx -v $(pwd)/data:/app/data travel-agent
#   docker compose up -d   # 推荐使用 docker-compose
# =============================================================================

# ---- Stage 1: Build ----
FROM python:3.11-slim AS builder

WORKDIR /app

# Editable install needs the package source to be present during build.
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -e ".[keyword,reranker,local-embeddings,observability,dev]"

# ---- Stage 2: Runtime ----
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

WORKDIR /app

# 从 builder 复制已安装的包
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# 复制应用代码
COPY pyproject.toml README.md ./
COPY src ./src
COPY docs ./docs
COPY tests ./tests

# 创建数据目录
RUN mkdir -p /app/data/chroma

# 默认显示帮助
CMD ["python", "-m", "travel_agent.rag.cli", "--help"]
