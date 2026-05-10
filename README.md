# travel-agent

纯 RAG 旅行目的地知识库项目。

本项目当前只做 RAG，不接入 LLM，不做 Agent 工作流，不做路线规划生成。系统负责导入目的地知识库文档（支持 `md` / `markdown` / `txt` / `pdf`）、切片、向量化、写入本地 Chroma 向量数据库，并支持检索、按目的地过滤、抽取式问答和召回质量评测。

## 项目边界

当前阶段包含：

- 文档导入
- 文本切片
- Embedding 向量化
- Chroma 本地向量库
- Retriever 检索
- destination / section / travel_type / season 元数据过滤
- 抽取式问答
- CLI 测试
- Python 模块调用
- 召回质量评测
- 增量重建已扫描文档的索引 chunk，并维护 ingest manifest
- 面向后续 Agent 节点的结构化 evidence 输出
- 纯 RAG 检索 trace、延迟统计和离线 eval 命令

当前阶段不包含：

- 不调用 ChatGPT / ChatOpenAI / 任何聊天 LLM
- 不生成完整旅行路线
- 不做 LangGraph Agent 工作流
- 不做天气、地图、预算、拥挤风险工具调用
- 不做 Langfuse / LangSmith 监控

说明：通义千问、OpenAI Embedding 和 sentence-transformers 都只是向量化模型，不是聊天 LLM。没有 API Key 或本地模型时可以使用 `LocalHashEmbeddings` 跑 demo 和测试。

## 已实现能力

- 使用 LangChain `DocumentLoader` 导入 Markdown、纯文本和 PDF 文档。
- 使用 LangChain `RecursiveCharacterTextSplitter` 进行文本切片。
- 支持按 Markdown 二级标题切分业务 section，并写入 chunk metadata。
- 使用 LangChain `Embedding` 抽象，默认优先使用通义千问 `text-embedding-v4`，并支持 OpenAI Embedding。
- 提供 `LocalHashEmbeddings` 本地测试 / demo embedding，方便无 API Key 时运行；它不代表真实语义召回质量。
- 支持可选的 `sentence-transformers` 本地多语言 embedding provider，默认本地真实模型为 `BAAI/bge-m3`。
- 使用 Chroma 作为本地持久化向量数据库。
- 提供 invoke 兼容的 `RagRetriever`，复用项目的过滤、fallback 和 rerank 逻辑。
- 支持 `vector`、`keyword`、`hybrid` 三种检索模式。
- `keyword` 模式使用进程内常驻 BM25 索引，避免每次检索重建关键词索引。
- `hybrid` 模式使用向量检索和 BM25 关键词检索，并通过可配置权重的 RRF 融合排序。
- 使用纯本地 `KeywordOverlapReranker` 做确定性 rerank，不调用 LLM 或外部模型。
- 使用 Typer 构建 CLI。
- 使用 Rich 美化终端输出。
- 使用 `.env` + `pydantic-settings` 管理配置。
- 使用 YAML front matter + Pydantic 校验目的地文档 metadata schema。
- 导入时为文档生成 `document_id` 和 `document_hash`，并替换同源旧 chunk，避免重复索引。
- 支持 `--incremental`：基于 manifest 跳过未变化文档。
- 支持按 `destination`、`section`、`travel_type`、`season` 元数据过滤检索结果。
- 将季节拆为 `season_spring` / `season_summer` / `season_autumn` / `season_winter` 字段，便于过滤下推。
- 检索结果返回 `content`、`source`、`destination`、`score` 和完整 chunk metadata。
- 提供 `retrieve_evidence` 结构化证据接口，返回召回结果、query analysis、confidence 和纯 RAG trace 指标，便于后续接入 LangGraph 节点。
- 提供 `travel-rag eval` 离线评测命令，输出 recall、MRR、precision、nDCG、keyword hit rate、metadata accuracy、empty-result 和 latency。
- 提供 `ask` 抽取式问答：只基于召回 chunk 整理答案，不调用 LLM。

## 快速开始

使用本地 embedding 运行：

```powershell
conda run -n Agent cmd /c "set PYTHONPATH=src&& python -m travel_agent.rag.cli reset --embedding-provider local --yes"
conda run -n Agent cmd /c "set PYTHONPATH=src&& python -m travel_agent.rag.cli ingest docs\destinations --embedding-provider local"
conda run -n Agent cmd /c "set PYTHONPATH=src&& python -m travel_agent.rag.cli query ""杭州灵隐寺周末拥挤吗？"" --destination Hangzhou --embedding-provider local --top-k 3"
conda run -n Agent cmd /c "set PYTHONPATH=src&& python -m travel_agent.rag.cli ask ""杭州灵隐寺周末拥挤吗？"" --destination Hangzhou --embedding-provider local --top-k 3"
```

修改 `docs/destinations/` 下的知识库文档后，重新执行 `ingest` 即可替换本次扫描到的同源旧 chunk；不需要为了单个文档变化重置整个库。已经启动的 `interactive` 会话不会自动加载新文档或新代码，需输入 `q` 退出后重新启动。

如果已安装为 editable package：

```powershell
conda run -n Agent python -m pip install -e .
travel-rag ingest docs\destinations --embedding-provider local
travel-rag query "带老人去杭州三天怎么安排？" --destination Hangzhou --embedding-provider local
travel-rag ask "杭州灵隐寺周末拥挤吗？" --destination Hangzhou --embedding-provider local
```

## 使用真实 Embedding

默认推荐使用通义千问 `text-embedding-v4`。复制 `.env.example` 为 `.env`，并配置：

```text
DASHSCOPE_API_KEY=你的阿里云百炼 API Key
TRAVEL_RAG_EMBEDDING_PROVIDER=auto
TRAVEL_RAG_QWEN_EMBEDDING_MODEL=text-embedding-v4
TRAVEL_RAG_QWEN_EMBEDDING_DIMENSIONS=1024
TRAVEL_RAG_QWEN_EMBEDDING_BATCH_SIZE=10
```

当 `TRAVEL_RAG_EMBEDDING_PROVIDER=auto` 且存在 `DASHSCOPE_API_KEY` 时，系统通过 OpenAI 兼容接口使用通义千问 Embedding；如果没有通义 Key 但存在 `OPENAI_API_KEY`，则使用 OpenAI Embedding；都没有时回退到 `LocalHashEmbeddings`。

`text-embedding-v4` 的批量请求默认按 10 条一批发送，避免超过百炼接口的 batch 限制。

验证通义千问 `text-embedding-v4`：

```powershell
conda run -n Agent cmd /c "set PYTHONPATH=src&& python -m travel_agent.rag.cli verify-embedding --embedding-provider qwen --json"
```

优先验证本地 `BAAI/bge-m3`：

```powershell
conda run -n Agent python -m pip install -e ".[local-embeddings]"
conda run -n Agent cmd /c "set PYTHONPATH=src&& python -m travel_agent.rag.cli verify-embedding --embedding-provider sentence-transformers --embedding-model BAAI/bge-m3 --json"
conda run -n Agent cmd /c "set PYTHONPATH=src&& python -m travel_agent.rag.cli eval --embedding-provider sentence-transformers --retrieval-mode hybrid --json"
```

可选验证 OpenAI Embedding：

复制 `.env.example` 为 `.env`，并配置：

```text
OPENAI_API_KEY=你的 OpenAI API Key
TRAVEL_RAG_EMBEDDING_PROVIDER=openai
TRAVEL_RAG_OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

```powershell
conda run -n Agent cmd /c "set PYTHONPATH=src&& python -m travel_agent.rag.cli verify-embedding --embedding-provider openai --json"
travel-rag ingest docs\destinations --embedding-provider openai
travel-rag query "东京亲子游需要注意什么？" --destination Tokyo --embedding-provider openai
```

## 知识库文档格式

当前 `ingest` 支持以下文件类型：

- `.md` / `.markdown`：按 Markdown 正文导入，支持 YAML front matter 和二级标题 section 识别。
- `.txt`：按 UTF-8 纯文本导入，可选 YAML front matter；未提供 `destination` 时使用文件名作为目的地。
- `.pdf`：通过 LangChain `PyPDFLoader` 抽取文本导入；未提供 `destination` 时使用文件名作为目的地。

推荐在 Markdown 或 txt 顶部写入 front matter。当前 RAG 元数据 schema 至少包含：

```markdown
---
destination: Hangzhou
city: Hangzhou
country: China
travel_type: family_free_independent
season: spring,autumn
source_type: destination_guide
updated_at: 2026-05-08
language: zh
last_verified_at: 2026-05-08
license: internal_demo
geo_area: Hangzhou urban area
price_level: mid
suitable_for: family,independent,elderly,children
poi_names: West Lake,Lingyin Temple,Longjing Village,Grand Canal,Liangzhu
---

# 杭州家庭自由行目的地知识

杭州适合第一次自由行、家庭出游和带老人慢节奏旅行。

## 概览

杭州适合家庭出游、亲子周末和带老人慢节奏旅行。

## 适合人群

带老人旅行适合选择步行量可控、休息点密集的西湖线。

## 交通

西湖周边节假日拥堵明显，建议优先使用地铁、公交和步行。

## 玩法

第一天可安排西湖湖滨、断桥、白堤、苏堤和雷峰塔。

## 预算

经济型旅行每日餐饮约 80 到 150 元。

## 住宿

湖滨和武林商圈适合第一次来杭州，餐饮、地铁和夜间活动方便。

## 餐饮

杭州餐饮以杭帮菜、面馆、点心和茶饮为主。

## 拥挤风险

节假日西湖断桥、雷峰塔和灵隐寺拥挤风险较高。

## 天气风险

春季和梅雨季雨水较多，需要准备防滑鞋、雨具和可替换袜子。

## 备选方案

若西湖人流过高，可改走浴鹄湾、茅家埠、京杭大运河或良渚古城遗址公园。
```

其中 `destination` 会作为检索过滤字段。查询时可以使用：

```powershell
travel-rag query "灵隐寺周末拥挤吗？" --destination Hangzhou --embedding-provider local
travel-rag query "杭州预算怎么安排？" --destination Hangzhou --section budget --travel-type family_free_independent --season autumn --embedding-provider local
```

也可以在导入时统一指定目的地：

```powershell
travel-rag ingest docs\destinations --destination Hangzhou --embedding-provider local
travel-rag ingest docs\destinations --embedding-provider local --incremental
```

导入 Markdown 时会根据二级标题推导 chunk 的 `section`；`txt` 和 `pdf` 默认进入 `overview`：

- `## 概览` -> `overview`
- `## 适合人群` -> `audience`
- `## 交通` -> `traffic`
- `## 玩法` -> `itinerary`
- `## 预算` -> `budget`
- `## 住宿` -> `lodging`
- `## 餐饮` -> `dining`
- `## 拥挤风险` -> `crowd_risk`
- `## 天气风险` -> `weather_risk`
- `## 备选方案` -> `alternatives`
- `## 风险提醒` -> `risk`

每个 chunk metadata 会包含：

`destination`、`city`、`country`、`travel_type`、`season`、`source_type`、`updated_at`、`language`、`source_url`、`license`、`last_verified_at`、`poi_names`、`geo_area`、`price_level`、`suitable_for`、`season_spring`、`season_summer`、`season_autumn`、`season_winter`、`document_id`、`document_hash`、`file_type`、`section`、`section_title`、`title`、`source`、`chunk_index`、`chunk_id`、`start_index`

检索支持的业务 metadata filter：

- `destination`：目的地过滤，例如 `Hangzhou`、`Tokyo`、`Suzhou`。
- `section`：二级标题推导出的业务段落，例如 `traffic`、`budget`、`crowd_risk`。
- `travel_type`：旅行类型，例如 `family_free_independent`。
- `season`：季节标签，例如 `spring`、`autumn`。导入时会派生季节 flag 字段用于过滤下推。
- `poi_names` / `geo_area` / `price_level` / `suitable_for`：旅行领域结构化标签，方便后续路线规划 Agent 做实体约束和偏好过滤。

兼容说明：当前普通检索路径会尽量把 `destination`、显式 `section`、`travel_type` 和季节 flag 下推到 Chroma metadata filter；同时保留 Python 侧后过滤作为兼容保护。`RagRetriever` 入口也会复用同一套检索 fallback，而不是直接裸用 Chroma retriever。

如果没有显式传入 `--section`，系统会根据问题中的常见业务词自动推断 section。例如包含“交通、地铁、机场、高铁、换乘、怎么去”的问题会优先检索 `traffic`；包含“预算、费用、门票”的问题会优先检索 `budget`；包含“拥挤、排队、人多”的问题会优先检索 `crowd_risk`。

## 内置目的地数据集

`docs/destinations/` 当前包含 8 个目的地文档，均使用统一 front matter schema 和二级标题 section 结构：

- `hangzhou.md`：杭州家庭自由行、亲子、老人慢游、拥挤和雨天备选。
- `tokyo_family.md`：东京亲子自由行、迪士尼、换乘、酒店和排队风险。
- `suzhou.md`：苏州园林古城、老人慢游、亲子研学、水乡备选。
- `dali.md`：大理洱海慢旅行、亲子自然体验、包车和天气风险。
- `changsha.md`：长沙周末美食、亲子城市游、夜间拥挤和雨天备选。
- `paris.md`：巴黎家庭自由行、博物馆、预约排队、安全和天气风险。
- `chengdu.md`：成都熊猫亲子、美食自由行、老人茶馆慢游和周边备选。
- `beijing.md`：北京亲子文化自由行、老人慢游、预约安检、长城和雨天备选。

## CLI 命令

```powershell
travel-rag ingest <path>      # 导入 md / txt / pdf 文件或目录
travel-rag query <question>   # 检索相关 chunk
travel-rag ask <question>     # 基于召回 chunk 做抽取式问答
travel-rag interactive        # 进入纯 RAG 交互式查询
travel-rag stats              # 查看 Chroma collection 统计信息
travel-rag verify-embedding   # 验证 embedding provider 是否可用
travel-rag eval               # 运行纯 RAG 离线召回评测
travel-rag reset --yes        # 清空本地 Chroma 索引
```

`query` 和 `ask` 支持业务 metadata 过滤参数：

```powershell
travel-rag query "苏州预算怎么安排？" --destination Suzhou --section budget --travel-type family_free_independent --season autumn --embedding-provider local
travel-rag ask "东京亲子游拥挤风险是什么？" --destination Tokyo --section crowd_risk --travel-type family_free_independent --embedding-provider local
travel-rag query "杭州预算门票多少钱？" --destination Hangzhou --section budget --retrieval-mode hybrid --embedding-provider local
```

默认检索模式是 `hybrid`。`--retrieval-mode` 可显式覆盖，可选：

- `vector`：向量检索模式；当前 Windows / Chroma 组合下如果向量索引短暂返回空结果，会用常驻 BM25 结果兜底。
- `keyword`：基于已 ingest chunk 的轻量 BM25 关键词检索，适合中文短 query、预算、交通、风险等关键词明显的问题。
- `hybrid`：同时执行 vector 和 keyword 检索，用 RRF 融合排序后返回。

RRF 默认公式为 `score += weight / (60 + rank)`，同一个 chunk 会按 `chunk_id` 去重并累加来自不同检索器的排名贡献。`TRAVEL_RAG_RRF_K`、`TRAVEL_RAG_VECTOR_WEIGHT`、`TRAVEL_RAG_KEYWORD_WEIGHT` 可调整融合行为。

`query` 返回检索结果：

- `content`：检索到的 chunk 内容
- `source`：来源知识库文件
- `destination`：目的地元数据
- `section`：业务段落 metadata，CLI 表格会单独展示
- `score`：相似度分数
- `metadata`：完整 chunk metadata

`ask` 返回抽取式答案：

- 先召回 top-k chunk
- 从 chunk 中抽取相关句子
- 拼成可读答案，每条信息会带上 `[中文目的地 / section_title]` 标签，例如 `[长沙 / 概览]`，便于区分跨目的地推荐
- 附带来源文档
- 不调用 LLM

`eval` 会临时构建评测索引并运行纯检索指标：

```powershell
travel-rag eval --embedding-provider local --retrieval-mode hybrid --json
```

输出包括 `recall_at_k`、`mrr_at_k`、`precision_at_k`、`ndcg_at_k`、`keyword_hit_rate_at_k`、`metadata_filter_accuracy`、`empty_result_rate`、`expected_empty_accuracy` 和 `avg_latency_ms`。

`eval` 默认启用质量门槛，低于阈值时命令会以非 0 状态退出，适合放进 CI：

- `recall_at_k >= 0.95`
- `mrr_at_k >= 0.90`
- `keyword_hit_rate_at_k >= 0.90`
- `metadata_filter_accuracy >= 1.00`
- `expected_empty_accuracy >= 0.50`

可通过 `--min-recall`、`--min-mrr`、`--min-keyword-hit-rate`、`--min-metadata-accuracy`、`--min-expected-empty-accuracy` 调整门槛；临时观察指标时可加 `--no-quality-gate`。

`interactive` 是连续提问模式，本质上仍然是纯 RAG 检索和抽取式回答，不接入 LLM：

```powershell
conda run -n Agent cmd /c "set PYTHONPATH=src&& python -m travel_agent.rag.cli interactive --embedding-provider local --top-k 3"
```

如果当前终端已经显示 `(Agent)`，说明 Conda 环境已激活，推荐直接运行：

```powershell
$env:PYTHONPATH="src"
python -m travel_agent.rag.cli interactive --embedding-provider local --top-k 3
```

进入后直接输入问题：

```text
请输入问题: 东京周末拥挤吗？
请输入问题: 杭州三天慢节奏怎么玩？
请输入问题: q
```

## 作为 Python 模块调用

推荐外部代码通过 `travel_agent.rag` 或 `travel_agent.rag.api` 调用 RAG 能力。

### 使用 `TravelRag` 客户端

```python
from travel_agent.rag import TravelRag

rag = TravelRag.create(
    persist_dir="data/chroma",
    embedding_provider="local",
)

rag.reset()
rag.ingest("docs/destinations")

results = rag.search(
    "杭州预算怎么安排？",
    destination="Hangzhou",
    section="budget",
    travel_type="family_free_independent",
    season="autumn",
    top_k=3,
)

for item in results:
    print(item.content)
    print(item.source, item.destination, item.metadata["section"], item.score)

answer = rag.ask(
    "杭州灵隐寺周末拥挤吗？",
    destination="Hangzhou",
    section="crowd_risk",
    top_k=3,
)

print(answer.answer)
```

### 获取结构化 EvidenceBundle

`retrieve_evidence` 不调用 LLM，只返回召回 chunk 和纯 RAG trace，适合作为后续 LangGraph 节点的输入。

```python
evidence = rag.retrieve_evidence(
    "北京长城天气不好时有什么备选方案？",
    section="alternatives",
    top_k=3,
    retrieval_mode="hybrid",
)

print(evidence.trace)
print(evidence.query_analysis)
print(evidence.confidence)
for item in evidence.results:
    print(item.source, item.metadata["section"], item.content)
```

`RetrievalTrace` 会记录一次检索的关键可观测信息：

- `metadata_filters`：下推到 retriever 的过滤条件，以及 Python 侧 post-filter 条件。
- `vector_hits`：向量检索阶段命中的 chunk 摘要。
- `keyword_hits`：BM25 检索阶段命中的 chunk 摘要。
- `fused_hits`：RRF 融合后的候选 chunk 摘要。
- `reranked_hits`：本地 reranker 和 `min_score` 过滤后的最终 chunk 摘要。
- `empty_result_reason`：空结果原因，例如 `empty_collection`、`no_candidates_from_retrievers`、`metadata_filters_removed_all`、`rerank_or_min_score_removed_all`。

每个 hit 摘要包含 `rank`、`source`、`section` 和 `score`，方便排查某次召回为什么命中或为什么被过滤掉。

### 使用一次性函数

```python
from travel_agent.rag import (
    answer_destination_question,
    ingest_destination_documents,
    retrieve_destination_evidence,
    search_destination_knowledge,
)

ingest_destination_documents(
    "docs/destinations",
    embedding_provider="local",
)

results = search_destination_knowledge(
    "东京亲子游怎么安排？",
    destination="Tokyo",
    section="itinerary",
    travel_type="family_free_independent",
    top_k=3,
    embedding_provider="local",
    retrieval_mode="hybrid",
)

answer = answer_destination_question(
    "杭州灵隐寺周末拥挤吗？",
    destination="Hangzhou",
    section="crowd_risk",
    embedding_provider="local",
)

print(answer.answer)

evidence = retrieve_destination_evidence(
    "巴黎卢浮宫排队太久可以改去哪？",
    section="alternatives",
    embedding_provider="local",
)
```

### 获取 LangChain Retriever

```python
from travel_agent.rag import get_destination_retriever

retriever = get_destination_retriever(
    destination="Hangzhou",
    section="budget",
    top_k=5,
    embedding_provider="local",
)

docs = retriever.invoke("适合老人游玩的杭州路线")
```

`RagRetriever` 提供 `invoke()` 兼容接口，并使用项目内置的 metadata filter、BM25 fallback 和本地 reranker；它不是裸 Chroma retriever。

## 配置项

配置通过 `.env` 和环境变量读取，前缀为 `TRAVEL_RAG_`。

```text
TRAVEL_RAG_PERSIST_DIR=data/chroma
TRAVEL_RAG_COLLECTION_NAME=travel_destinations
TRAVEL_RAG_EMBEDDING_PROVIDER=auto
TRAVEL_RAG_QWEN_EMBEDDING_MODEL=text-embedding-v4
TRAVEL_RAG_QWEN_EMBEDDING_DIMENSIONS=1024
TRAVEL_RAG_QWEN_EMBEDDING_BATCH_SIZE=10
TRAVEL_RAG_QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
TRAVEL_RAG_OPENAI_EMBEDDING_MODEL=text-embedding-3-small
TRAVEL_RAG_SENTENCE_TRANSFORMERS_MODEL=BAAI/bge-m3
TRAVEL_RAG_LOCAL_EMBEDDING_DIMENSIONS=512
TRAVEL_RAG_CHUNK_SIZE=800
TRAVEL_RAG_CHUNK_OVERLAP=120
TRAVEL_RAG_DEFAULT_TOP_K=5
TRAVEL_RAG_RETRIEVAL_MODE=hybrid
TRAVEL_RAG_RETRIEVAL_CANDIDATE_MULTIPLIER=5
TRAVEL_RAG_RRF_K=60
TRAVEL_RAG_VECTOR_WEIGHT=1.0
TRAVEL_RAG_KEYWORD_WEIGHT=1.0
TRAVEL_RAG_MIN_SCORE=0.0
DASHSCOPE_API_KEY=
OPENAI_API_KEY=
```

`TRAVEL_RAG_RETRIEVAL_MODE` 默认推荐使用 `hybrid`，也可设为 `vector` 或 `keyword`。CLI 和 Python API 中显式传入的 `retrieval_mode` 会覆盖配置默认值。

Embedding provider 说明：

- `local`：确定性 hash embedding，只用于单测、CI、离线 demo 和无 API Key 演示，不代表真实语义召回质量。
- `qwen` / `dashscope`：通过 OpenAI 兼容接口调用通义千问 Embedding，默认模型为 `text-embedding-v4`。
- `openai`：只使用 OpenAI Embedding，不调用聊天 LLM。
- `sentence-transformers`：可选本地多语言 embedding，需要安装 `.[local-embeddings]`。
- `auto`：优先使用 `DASHSCOPE_API_KEY` 对应的通义千问 Embedding；没有通义 Key 但有 `OPENAI_API_KEY` 时使用 OpenAI Embedding；都没有时回退到 `local`。

## 项目结构

```text
src/travel_agent/rag/
  api.py                  # 对外 Python 调用接口
  cli.py                  # Typer + Rich CLI
  service.py              # RAG 应用服务：导入、检索、抽取式问答
  evaluation.py           # 纯 RAG 离线评测入口
  manifest.py             # ingest manifest 和 collection version
  loaders.py              # LangChain md / txt / pdf Loader
  splitters.py            # LangChain TextSplitter 工厂
  embeddings.py           # 通义千问 / OpenAI / sentence-transformers / 本地 fallback
  keyword.py              # 常驻 BM25 关键词索引
  rerankers.py            # Reranker 协议和纯本地 KeywordOverlapReranker
  vector_store.py         # Chroma VectorStore 辅助函数
  langchain_adapters.py   # LangChain Document 适配工具
  models.py               # 响应模型
  config.py               # pydantic-settings 配置
tests/                    # 单元测试和召回质量评测
docs/destinations/        # 示例目的地 Markdown 文档
.github/workflows/ci.yml  # GitHub Actions CI
Dockerfile                # 容器化运行示例
Makefile                  # test / lint / compile / eval 快捷命令
```

## 测试

使用 Conda `Agent` 环境运行：

```powershell
conda run -n Agent cmd /c "set PYTHONPATH=src&& python -m pytest tests -q -p no:cacheprovider"
conda run -n Agent python -m ruff check src tests
conda run -n Agent python -m compileall src tests
```

## RAG 召回质量评测

```powershell
conda run -n Agent cmd /c "set PYTHONPATH=src&& python -m pytest tests\test_recall_quality.py -q -p no:cacheprovider"
conda run -n Agent cmd /c "set PYTHONPATH=src&& python -m travel_agent.rag.cli eval --embedding-provider local --retrieval-mode hybrid --json"
```

评测入口：[tests/test_recall_quality.py](tests/test_recall_quality.py)

结构化评测数据：[tests/fixtures/rag_eval_cases.jsonl](tests/fixtures/rag_eval_cases.jsonl)

每个 case 包含：

- `query`
- `destination`（可选；不传时用于测试目的地自动推断）
- `section`（可选）
- `travel_type`（可选）
- `season`（可选）
- `expected_source`
- `expected_keywords`
- `expected_empty`（可选，用于无答案 / hard negative）
- `hard_negative`（可选，用于容易误召回的混淆问题）

当前评测覆盖：

- `recall@3`：top-3 结果是否命中预期来源文档。
- `MRR@3`：预期来源文档在 top-3 中的倒数排名均值。
- `keyword_hit_rate@3`：top-3 内容中是否覆盖预期关键词。
- `metadata_filter_accuracy`：返回结果是否满足 case 中声明的 `destination`、`section`、`travel_type`、`season` 过滤条件。
- `precision_at_k`、`ndcg_at_k`、`empty_result_rate`、`expected_empty_accuracy` 和 `avg_latency_ms`：由 `travel-rag eval` 输出。

`travel-rag eval` 默认质量门槛与 CI 建议一致：

- `recall_at_k >= 0.95`
- `mrr_at_k >= 0.90`
- `keyword_hit_rate_at_k >= 0.90`
- `metadata_filter_accuracy >= 1.00`
- `expected_empty_accuracy >= 0.50`

当前 fixture 包含 18 个 case，覆盖：

- 杭州交通拥挤、预算、拥挤风险、老人慢节奏、雨天备选。
- 东京亲子玩法、酒店切换、交通换乘、迪士尼拥挤风险、雨天备选、老人慢节奏、预算。
- 北京、巴黎、成都的不显式目的地推断与备选 / 拥挤风险检索。
- 杭州迪士尼、东京西湖等 hard negative，以及无答案目的地问题。

当前本地 `LocalHashEmbeddings` + `hybrid` 评测结果：

- `recall_at_k = 1.000`
- `mrr_at_k = 1.000`
- `precision_at_k = 0.333`
- `ndcg_at_k = 1.000`
- `keyword_hit_rate_at_k = 1.000`
- `metadata_filter_accuracy = 1.000`
- `expected_empty_accuracy = 1.000`

## Windows / Chroma 兼容说明

在当前 Windows + Conda 环境中，ChromaDB 1.x 的 native upsert 路径会触发 `python.exe` access violation 弹窗；ChromaDB 0.4.x 又会在 import 阶段尝试初始化默认 ONNX embedding。

本项目已做兼容处理：

- `chromadb` 固定为 `>=0.4.22,<0.5`
- `numpy` 固定为 `>=1.24,<2`
- 项目自行提供 LangChain embedding，因此在 `vector_store.py` 中禁用了 Chroma 未使用的默认 ONNX embedding
- vector 检索在该环境返回空结果时，会使用常驻 BM25 索引兜底，避免 CLI 查询直接空掉
- 建议不要并行运行多个会写入 Chroma 的 pytest / eval 进程；Windows 本地 Chroma 文件锁和 native HNSW 较敏感
- 本地 demo 使用 `--embedding-provider local` 即可无 API Key 运行

## 清单完成情况

- 增量导入：支持按 `source` / `document_id` 删除旧 chunk，支持 `document_hash` 去重，维护 `ingest_manifest.json`，并提供 `travel-rag ingest --incremental`。
- Embedding 分层：`auto` 默认优先通义千问 `text-embedding-v4`；`local` 明确为测试 / demo；`openai` 只用于 embedding；可选 `sentence-transformers` 本地多语言 embedding；`verify-embedding` 可做 provider smoke check。
- 评测扩展：fixture 覆盖自动目的地推断、hard negative、无答案问题，并新增 `travel-rag eval` 指标报告。
- Metadata schema：使用 YAML front matter + Pydantic 校验，补充来源、许可、验证时间和旅行领域结构化字段。
- 检索优化：支持 metadata filter 下推、季节 flag 字段、常驻 BM25 缓存、可配置 RRF 参数和本地 reranker。
- Evidence 接口：`retrieve_evidence()` 返回 evidence、query analysis、confidence 和 trace，可供后续 LangGraph 节点消费。
- 可观测性基础：`RetrievalTrace` 包含 `trace_id`、provider、collection version、latency、metadata filters、vector / keyword / fused / reranked hits、空结果原因和平均分。
- 工程化：补充 ruff / pytest / compileall 验证、GitHub Actions、Makefile、Dockerfile、`.env.example` 和缓存忽略规则。
- 旅行领域结构化：示例目的地 front matter 已包含 `poi_names`、`geo_area`、`price_level`、`suitable_for` 等字段。
