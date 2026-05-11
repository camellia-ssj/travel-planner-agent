# travel-agent Makefile
# 默认使用 Conda Agent 环境运行所有命令
CONDA_ENV := Agent
PYTEST := conda run -n $(CONDA_ENV) python -m pytest
RUFF := conda run -n $(CONDA_ENV) python -m ruff
COMPILEALL := conda run -n $(CONDA_ENV) python -m compileall
RAG_CLI := conda run -n $(CONDA_ENV) python -m travel_agent.rag.cli
AGENT_CLI := conda run -n $(CONDA_ENV) python -m travel_agent.agent.cli
PIP := conda run -n $(CONDA_ENV) python -m pip

# Export PYTHONPATH for all commands
export PYTHONPATH := src

.PHONY: help install install-dev install-all test test-rag test-agent test-tools lint compile eval-rag eval-agent clean reset demo check

help: ## 显示所有可用命令
	@echo "travel-agent Makefile"
	@echo ""
	@echo "开发命令:"
	@echo "  make install       安装核心依赖"
	@echo "  make install-dev   安装开发依赖"
	@echo "  make install-all   安装全部可选依赖"
	@echo ""
	@echo "测试命令:"
	@echo "  make test          运行全量测试"
	@echo "  make test-rag      运行 RAG 测试"
	@echo "  make test-agent    运行 Agent 测试"
	@echo "  make test-tools    运行工具函数测试"
	@echo ""
	@echo "质量命令:"
	@echo "  make lint          代码风格检查 (ruff)"
	@echo "  make compile       编译验证"
	@echo "  make check         lint + compile + test"
	@echo ""
	@echo "评测命令:"
	@echo "  make eval-rag      运行 RAG 离线评测"
	@echo "  make eval-agent    运行 Agent 离线评测"
	@echo ""
	@echo "数据命令:"
	@echo "  make reset         清空向量数据库"
	@echo "  make ingest        导入知识库文档"
	@echo ""
	@echo "Demo 命令:"
	@echo "  make demo          一键运行完整 demo (reset + ingest + query)"
	@echo ""
	@echo "Docker 命令:"
	@echo "  make docker-build  构建 Docker 镜像"
	@echo "  make docker-up     启动 docker-compose 服务"
	@echo "  make docker-down   停止 docker-compose 服务"
	@echo ""

# ---- 安装 ----
install: ## 安装核心依赖
	$(PIP) install -e .

install-dev: ## 安装核心 + 开发依赖
	$(PIP) install -e ".[dev]"

install-all: ## 安装全部依赖 (含 keyword, reranker, local-embeddings, observability, dev)
	$(PIP) install -e ".[keyword,reranker,local-embeddings,observability,dev]"

# ---- 测试 ----
test: ## 运行全量测试
	$(PYTEST) tests -q -p no:cacheprovider

test-rag: ## 运行 RAG 测试
	$(PYTEST) tests/test_rag_pipeline.py tests/test_recall_quality.py -q -p no:cacheprovider

test-agent: ## 运行 Agent 测试
	$(PYTEST) tests/test_agent_graph.py -q -p no:cacheprovider

test-tools: ## 运行工具函数测试
	$(PYTEST) tests/test_tools.py -q -p no:cacheprovider

# ---- 代码质量 ----
lint: ## 代码风格检查
	$(RUFF) check src tests

compile: ## 编译验证
	$(COMPILEALL) src tests

check: lint compile test ## lint + compile + test 全量检查

# ---- 评测 ----
eval-rag: ## RAG 离线评测
	$(RAG_CLI) eval --embedding-provider local --retrieval-mode hybrid --json

eval-agent: ## Agent 离线评测
	$(AGENT_CLI) eval --json --verbose

# ---- 数据 ----
reset: ## 清空向量数据库 (需确认)
	$(RAG_CLI) reset --yes

ingest: ## 导入知识库文档
	$(RAG_CLI) ingest docs/destinations --embedding-provider local

# ---- Demo ----
demo: reset ingest ## 一键 Demo: reset + ingest + query
	@echo ""
	@echo "=== Demo: 检索测试 ==="
	$(RAG_CLI) query "杭州灵隐寺周末拥挤吗？" --destination Hangzhou --top-k 3
	@echo ""
	@echo "=== Demo: 抽取式问答 ==="
	$(RAG_CLI) ask "杭州灵隐寺周末拥挤吗？" --destination Hangzhou --top-k 3
	@echo ""
	@echo "=== Demo: Agent 规划 ==="
	$(AGENT_CLI) plan "我和父母去杭州玩3天，预算中等" --embedding-provider local

# ---- Docker ----
docker-build: ## 构建 Docker 镜像
	docker build -t travel-agent .

docker-up: ## 启动 docker-compose 服务
	docker compose up -d

docker-down: ## 停止 docker-compose 服务
	docker compose down

# ---- 清理 ----
clean: ## 清理临时文件
	@echo "清理 Python 缓存..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "Done."
