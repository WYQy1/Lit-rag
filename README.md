# Lit-rag

[![CI](https://github.com/WYQy1/Lit-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/WYQy1/Lit-rag/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Ruff](https://img.shields.io/badge/lint-ruff-261230)
![License](https://img.shields.io/badge/license-MIT-green)

文献检索问答（RAG）系统 —— 记录从「会调用大模型 API」到「能做出可用的文献问答系统」的完整过程。

- **作者**：WYQy1
- **定位**：AI Agent / 大模型应用开发方向的学习与作品集项目
- **技术栈**：Python 3.11+ · uv · numpy · PyMuPDF · pypdf · OpenAI SDK · pytest · ruff · GitHub Actions

---

## 这个项目解决什么问题

课题组积累的发光材料文献动辄几百上千篇，人工检索效率低、跨文献对比困难。本项目把 PDF 文献变成**可检索、可溯源**的知识库：给定一个问题，找出最相关的原文片段，并标明它出自哪篇文献、哪一页、哪一章节。

同时，这套流水线本身就是完整的 RAG 工程练习——解析、切分、向量化、检索四步，每一步都有真实的坑。

---

## 架构

```
                       PDF 文件
                          │
                          ▼
        ┌─────────────────────────────────────┐
        │ parsing                             │
        │   loader   PyMuPDF / pypdf 抽取文本  │
        │   cleaner  去页眉页脚 · 修断词 · 删页码│
        └──────────────────┬──────────────────┘
                           ▼
        ┌─────────────────────────────────────┐
        │ chunking                            │
        │   splitter  段落优先 + 章节识别       │
        │             + 超长段落按句切分        │
        │             + 同章节内相邻重叠        │
        └──────────────────┬──────────────────┘
                           ▼
        ┌─────────────────────────────────────┐
        │ embedding                           │
        │   offline  哈希向量化（离线，可测）   │
        │   openai   BGE-M3 / embedding-3 / … │
        └──────────────────┬──────────────────┘
                           ▼
        ┌─────────────────────────────────────┐
        │ store                               │
        │   memory   余弦检索 + 磁盘持久化      │
        └──────────────────┬──────────────────┘
                           ▼
              检索结果（带出处 · 页码 · 章节）
```

---

## 目录结构

```
Lit-rag/
├── src/lit_rag/               # 主包
│   ├── schema.py              # Document / Chunk / Hit 数据结构
│   ├── config.py              # 运行配置（统一从环境变量读取）
│   ├── pipeline.py            # 编排层：解析→切分→向量化→入库→检索
│   ├── parsing/
│   │   ├── loader.py          # PDF 加载（双后端 + 内容寻址 doc_id）
│   │   └── cleaner.py         # 文本清洗
│   ├── chunking/
│   │   └── splitter.py        # 结构感知切分
│   ├── embedding/
│   │   ├── base.py            # Embedder 抽象
│   │   ├── offline.py         # 离线哈希向量化（CI 用）
│   │   └── openai_compat.py   # OpenAI 兼容接口
│   └── store/
│       └── memory.py          # 内存向量库 + 持久化
├── scripts/
│   ├── ingest.py              # CLI：构建索引
│   ├── search.py              # CLI：检索
│   └── make_sample_pdf.py     # 生成测试用样本 PDF
├── samples/                   # 样本数据（可提交）
├── tests/                     # 143 个测试，全部离线可跑
├── hello-llm/                 # 早期学习脚本（API 调用、Prompt 实验）
├── .github/workflows/ci.yml   # CI：ruff + pytest（3 个 Python 版本）
├── .env.example               # 环境变量模板
└── pyproject.toml             # 依赖 / ruff / pytest 统一配置
```

---

## 快速开始

### 1. 安装

```powershell
pip install uv          # 若尚未安装
uv sync                 # 创建 .venv 并安装依赖
```

### 2. 配置密钥

```powershell
copy .env.example .env
# 编辑 .env 填入你的 Key
```

> **并不强制**：`LIT_RAG_EMBED_PROVIDER` 默认是 `offline`，用内置哈希向量化器，
> 无需任何 Key 就能跑通全流程（检索质量较低，仅用于验证链路）。

### 3. 构建索引

```powershell
# 用仓库自带样本试跑
uv run python scripts/ingest.py --input samples --output data/index

# 用自己的文献
uv run python scripts/ingest.py --input data/raw --output data/index
```

### 4. 检索

```powershell
uv run python scripts/search.py "Cr3+ 掺杂浓度对发射峰位有什么影响" --top-k 5
```

输出示例：

```
查询：concentration quenching and lifetime shortening
索引：data/index（9 个 chunk，1024 维）
embedder：offline-hashing(dim=1024, ngram=2-4)
------------------------------------------------------------
[1] score=0.3959  nir-phosphor-demo.pdf p.1 §3.2 Luminescence properties
    Under blue excitation all samples exhibited a broad emission band centred
    in the near-infrared region. With increasing Cr3+ concentration, the
    emission maximum shifted from 748 nm to 812 nm …
------------------------------------------------------------
共 3 条结果
```

---

## ⚠️ 关于向量化：DeepSeek 不提供 embeddings

这是一个必须提前知道的坑：**DeepSeek 的 API 只有对话模型，没有 embeddings 接口**。所以：

- 对话可以继续用 DeepSeek（`DEEPSEEK_API_KEY`）
- 但**向量化必须另找一家**，两者 Key 不通用

推荐方案（按性价比）：

| 服务商 | 模型 | base_url |
|---|---|---|
| 硅基流动 SiliconFlow | `BAAI/bge-m3`（有免费额度，中文好） | `https://api.siliconflow.cn/v1` |
| 智谱 AI | `embedding-3` | `https://open.bigmodel.cn/api/paas/v4` |
| 阿里云百炼 | `text-embedding-v4` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| OpenAI | `text-embedding-3-small` | `https://api.openai.com/v1` |

配置方式（`.env`）：

```ini
LIT_RAG_EMBED_PROVIDER=openai
LIT_RAG_EMBED_MODEL=BAAI/bge-m3
LIT_RAG_EMBED_BASE_URL=https://api.siliconflow.cn/v1
LIT_RAG_EMBED_API_KEY=sk-你的向量化Key
```

> **一致性检查**：`search.py` 会比对索引里记录的 embedder 与当前使用的 embedder，
> 不一致时会告警——用不同的模型建索引和查询，分数是没有意义的。

---

## 🔐 密钥安全

- 真实 Key **只放在 `.env`**，它已被 `.gitignore` 忽略
- 代码里一律 `os.getenv(...)`，绝不硬编码
- `config.Settings.describe()` 打印配置时自动脱敏
- `tests/test_smoke.py` 有一条**回归测试**专门检查 `.env` 是否仍被忽略——改坏忽略规则 CI 会立刻失败

> Key 一旦被提交过，删文件没用，必须去平台**吊销并重新生成**。

---

## 开发

```powershell
uv run ruff check .              # 代码检查
uv run ruff check . --fix        # 自动修复
uv run ruff format .             # 格式化
uv run pytest                    # 全部测试
uv run pytest --cov=src/lit_rag --cov-report=term-missing   # 覆盖率
uv run pytest -m integration     # 需要真实 API Key 的测试（默认跳过）
```

提交前跑一遍：

```powershell
uv run ruff check . ; uv run pytest
```

---

## 设计要点

几个刻意做的工程决策，也是面试时可以展开讲的点：

| 决策 | 原因 |
|---|---|
| **逐行扫描识别章节，而不是按空行切段** | PDF 抽取出来的文本几乎没有空行，一段话会被拆成很多物理行。按空行切分会导致章节标题被埋在段落中间，`section` 全部丢失 |
| **同段落物理行先拼回一行** | 否则句子切分失效。中文之间直接拼接，中英之间补空格，避免「这是 一 句话」和 `maximum.摘要` 两种粘连 |
| **切分时换章节强制断开，且不跨章节加重叠** | 否则上一章的内容会被标成新章节，引用信息失真 |
| **向量化用 `embed_text`（带章节标题前缀），存库用 `text`（纯正文）** | 检索受益于上下文，展示给用户时又保持干净 |
| **`doc_id` 用文件内容 SHA-1** | 天然去重、支持增量更新，同一份 PDF 重复入库得到相同 ID |
| **离线哈希 embedder** | 让整条流水线在无网络、无 Key 的环境下也能被测；CI 因此能跑真检索而非 mock |
| **哈希用 blake2b 而非内置 `hash()`** | Python 内置 `hash()` 对字符串加随机盐，跨进程不可复现 |
| **索引里记录 embedder 名称并做一致性校验** | 用不同模型建索引和查询是 RAG 最常见的静默 bug，分数全错却看不出来 |
| **样本 PDF 由脚本生成而非直接放论文** | 可复现、无版权风险，且刻意埋了页眉/页码/章节标题/参考文献等噪声 |

---

## 学习进度

- [x] 环境搭建：Python、uv、Git、SSH、GitHub Actions
- [x] 第一个大模型 API 调用（DeepSeek）
- [x] Prompt 实验：零样本 CoT / 少样本 CoT / 角色设定
- [x] 工程化：pyproject、ruff、pytest、CI
- [x] **PDF 解析与文本清洗**（双后端 + 页眉页脚/断词/页码处理）
- [x] **结构感知切分**（章节识别 + 段落优先 + 重叠）
- [x] **向量化抽象层**（离线 + OpenAI 兼容）
- [x] **向量检索与持久化**（余弦相似度 + 出处溯源）
- [ ] Rerank 重排序
- [ ] 检索质量评估（Recall@K、MRR、RAGAS）
- [ ] 带引用的问答生成（LLM 接入）
- [ ] 混合检索（BM25 + 向量）
- [ ] FastAPI 服务化
- [ ] 换成 Chroma / Milvus 等生产级向量库

---

## License

MIT
