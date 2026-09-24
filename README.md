# Lit-rag

[![CI](https://github.com/WYQy1/Lit-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/WYQy1/Lit-rag/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Ruff](https://img.shields.io/badge/lint-ruff-261230)
![License](https://img.shields.io/badge/license-MIT-green)

文献检索问答（RAG）系统 —— 记录从「会调用大模型 API」到「能做出可用的文献问答系统」的完整过程。

- **作者**：WYQy1
- **定位**：AI Agent / 大模型应用开发方向的学习与作品集项目
- **技术栈**：Python 3.11+ · OpenAI SDK · python-dotenv · pytest · ruff · GitHub Actions

---

## 项目目标

最终目标是把「发光材料 / 光学」领域的文献变成可检索、可问答、可溯源的知识库，同时用这个真实需求练出完整的 RAG 与 Agent 工程能力。

| 阶段 | 目标 | 状态 |
|---|---|---|
| W1 | 打通大模型 API 调用（DeepSeek） | ✅ 已完成 |
| W2 | Prompt 基础实验（零样本 / 少样本 CoT / 角色设定） | ✅ 已完成 |
| W3 | 工程化：依赖管理、lint、测试、CI | ✅ 已完成 |
| W4+ | 文档解析与切分 → Embedding → 向量检索 → Rerank → 带引用问答 | ⬜ 进行中 |

---

## 目录结构

```
Lit-rag/
├── .github/workflows/ci.yml   # GitHub Actions：lint + 测试
├── hello-llm/                 # 学习阶段脚本（API 调用、Prompt 实验）
│   ├── hello-llm.py           # 第一个大模型 API 调用
│   ├── prompt_lab.py          # Prompt 对比实验
│   └── .env                   # 本地密钥（已被 git 忽略，不提交）
├── tests/                     # pytest 测试
│   └── test_smoke.py          # 冒烟测试 + 密钥泄露回归测试
├── main.py                    # 项目入口
├── test.py / test2.py         # 早期临时脚本（未参与 lint）
├── .env.example               # 环境变量模板（可安全提交）
├── .gitattributes             # 统一换行与二进制处理
├── .gitignore                 # 忽略规则（含 .env）
└── pyproject.toml             # 依赖、ruff、pytest 统一配置
```

---

## 快速开始

### 方式一：uv（推荐）

```powershell
# 1. 安装 uv（若尚未安装）
pip install uv

# 2. 创建虚拟环境并安装依赖（含 dev 依赖）
uv sync

# 3. 配置密钥
copy .env.example .env
# 然后编辑 .env，填入你的真实 DEEPSEEK_API_KEY
```

### 方式二：pip + venv

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

### 运行

```powershell
# 运行项目入口
uv run python main.py

# 运行学习脚本
uv run python hello-llm/hello-llm.py
uv run python hello-llm/prompt_lab.py
```

> 在 PyCharm / VS Code 中，请把项目解释器指向 `.venv\Scripts\python.exe`。

---

## 🔐 密钥安全（重要）

- **真实 Key 只放在 `.env` 里**，`.env` 已被 `.gitignore` 忽略。
- 需要提交的模板是 `.env.example`，里面只放占位符。
- 代码中一律用 `os.getenv("DEEPSEEK_API_KEY")` 读取，**绝不硬编码**。
- `tests/test_smoke.py` 里有一条回归测试专门检查 `.env` 是否仍被忽略 —— 一旦有人改坏忽略规则，CI 会立刻失败。

> 如果 Key 曾经被提交过，仅删除文件是不够的，必须去平台**吊销并重新生成** Key。

---

## 开发

```powershell
# 代码检查
uv run ruff check .

# 自动修复
uv run ruff check . --fix

# 代码格式化
uv run ruff format .

# 运行测试
uv run pytest

# 带覆盖率
uv run pytest --cov=. --cov-report=term-missing
```

提交前建议跑一遍：

```powershell
uv run ruff check . ; uv run pytest
```

---

## 学习进度

- [x] 环境搭建：Python、uv、Git、SSH 密钥、GitHub 打通
- [x] 第一个大模型 API 调用（DeepSeek）
- [x] Prompt 实验：零样本 CoT / 少样本 CoT / 角色设定
- [x] 工程化：pyproject、ruff、pytest、GitHub Actions CI
- [ ] 文档解析与切分（PDF → 结构化文本）
- [ ] Embedding 与向量检索
- [ ] Rerank 与引用溯源
- [ ] 检索质量评估（RAGAS）
- [ ] FastAPI 服务化

---

## License

MIT
