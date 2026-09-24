"""项目核心数据结构。

统一使用 dataclass，便于序列化与调试，同时避免引入额外依赖。
所有对象都是「纯数据」，不承载行为逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Page:
    """PDF 的单页文本。"""

    number: int  # 1-based 页码
    text: str


@dataclass(slots=True)
class Document:
    """一份解析后的文档。"""

    doc_id: str
    source: str  # 原始文件路径（存字符串，便于 JSON 序列化）
    title: str = ""
    pages: list[Page] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """整篇文档的纯文本。"""
        return "\n\n".join(page.text for page in self.pages)

    @property
    def n_pages(self) -> int:
        return len(self.pages)

    @property
    def n_chars(self) -> int:
        return sum(len(page.text) for page in self.pages)


@dataclass(slots=True)
class Chunk:
    """切分后的文本块，是检索与引用的最小单位。"""

    chunk_id: str
    doc_id: str
    text: str
    index: int  # 在所属文档内的序号，从 0 开始
    page_start: int = 0
    page_end: int = 0
    section: str = ""  # 所属章节标题，供引用展示与过滤
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_chars(self) -> int:
        return len(self.text)

    @property
    def embed_text(self) -> str:
        """用于向量化的文本。

        在正文前拼接章节标题，让「2.1 Sample preparation」这类上下文
        也参与语义匹配 —— 这个小技巧对检索召回率提升明显。
        """
        if self.section:
            return f"{self.section}\n{self.text}"
        return self.text

    def citation(self) -> str:
        """生成人类可读的引用串，例如 ``paper.pdf p.3 §2.1``。"""
        parts = [str(self.metadata.get("source", self.doc_id))]
        if self.page_start:
            page = f"p.{self.page_start}"
            if self.page_end and self.page_end != self.page_start:
                page += f"-{self.page_end}"
            parts.append(page)
        if self.section:
            parts.append(f"§{self.section}")
        return " ".join(parts)


@dataclass(slots=True)
class Hit:
    """一次检索命中。"""

    chunk: Chunk
    score: float

    def __repr__(self) -> str:  # pragma: no cover - 仅用于调试输出
        preview = self.chunk.text[:40].replace("\n", " ")
        return f"Hit(score={self.score:.4f}, {preview!r}...)"
