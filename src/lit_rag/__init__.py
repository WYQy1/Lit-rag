"""Lit-rag：面向文献检索问答（RAG）的端到端流水线。

模块划分：

- ``lit_rag.schema``     核心数据结构（Document / Chunk / Hit）
- ``lit_rag.config``     运行配置（统一从环境变量读取）
- ``lit_rag.parsing``    PDF 解析与文本清洗
- ``lit_rag.chunking``   结构感知的文本切分
- ``lit_rag.embedding``  向量化（离线实现 + OpenAI 兼容实现）
- ``lit_rag.store``      向量存储与相似度检索
- ``lit_rag.pipeline``   把上面几步串起来的编排层
"""

from __future__ import annotations

__version__ = "0.2.0"

from lit_rag.schema import Chunk, Document, Hit, Page

__all__ = ["Chunk", "Document", "Hit", "Page", "__version__"]
