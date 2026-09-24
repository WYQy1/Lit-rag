"""编排层：把「解析 → 切分 → 向量化 → 入库 → 检索」串成一条流水线。

上层（CLI 脚本、将来的 FastAPI 服务）只需要调用这里的函数，
不必关心各模块的内部细节。
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

from lit_rag.chunking import split_documents
from lit_rag.config import Settings
from lit_rag.embedding import Embedder, create_embedder
from lit_rag.parsing import iter_pdf_paths, load_pdfs
from lit_rag.schema import Chunk, Document, Hit
from lit_rag.store import VectorStore

logger = logging.getLogger(__name__)


def resolve_pdf_paths(inputs: Iterable[str | Path]) -> list[Path]:
    """把「文件 + 目录」混合的输入展开为去重后的 PDF 路径列表。

    去重按绝对路径（Windows 下大小写不敏感），保证重复传入不会重复入库。
    """
    collected: list[Path] = []
    seen: set[str] = set()
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            candidates = iter_pdf_paths(path)
        elif path.is_file():
            candidates = [path] if path.suffix.lower() == ".pdf" else []
        else:
            logger.warning("路径不存在，已跳过：%s", path)
            continue
        for candidate in candidates:
            key = str(candidate.resolve()).lower()
            if key not in seen:
                seen.add(key)
                collected.append(candidate)
    return sorted(collected)


def load_documents(paths: Sequence[str | Path], *, settings: Settings) -> list[Document]:
    """解析一批 PDF。"""
    return load_pdfs(
        list(paths),
        backend=settings.pdf_backend,
        drop_references=settings.drop_references,
    )


def chunks_from_documents(docs: Sequence[Document], *, settings: Settings) -> list[Chunk]:
    """按配置切分文档。"""
    return split_documents(
        list(docs),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        min_chunk_size=settings.min_chunk_size,
    )


def index_chunks(chunks: Sequence[Chunk], embedder: Embedder) -> VectorStore:
    """把 chunk 向量化并装入新的索引。

    注意：送入 embedder 的是 :attr:`Chunk.embed_text`（带章节标题前缀），
    而存进索引的是 :attr:`Chunk.text`（纯正文），这样检索受益于上下文、
    展示给用户时又保持干净。
    """
    store = VectorStore()
    if not chunks:
        return store
    vectors = embedder.embed([chunk.embed_text for chunk in chunks])
    store.add(list(chunks), vectors)
    return store


def ingest(
    inputs: Iterable[str | Path],
    *,
    settings: Settings | None = None,
    embedder: Embedder | None = None,
) -> VectorStore:
    """从 PDF 文件/目录构建检索索引。

    Args:
        inputs: PDF 文件或目录路径。
        settings: 运行配置，默认从环境变量读取。
        embedder: 自定义向量化器，默认按配置创建。

    Returns:
        构建好的 :class:`~lit_rag.store.VectorStore`。
    """
    cfg = settings or Settings.from_env()
    encoder = embedder or create_embedder(cfg)

    paths = resolve_pdf_paths(inputs)
    if not paths:
        raise FileNotFoundError("没有找到任何 PDF 文件，请检查 --input 指向的路径")

    logger.info("发现 %d 个 PDF，使用 embedder: %s", len(paths), encoder.name)
    docs = load_documents(paths, settings=cfg)
    if not docs:
        raise RuntimeError("所有 PDF 都解析失败，请检查文件是否损坏或被加密")

    chunks = chunks_from_documents(docs, settings=cfg)
    logger.info(
        "解析成功 %d 篇，共 %d 字符，切出 %d 个 chunk",
        len(docs),
        sum(doc.n_chars for doc in docs),
        len(chunks),
    )

    store = index_chunks(chunks, encoder)
    logger.info("索引构建完成：%s", store)
    return store


def retrieve(
    query: str,
    store: VectorStore,
    embedder: Embedder,
    *,
    top_k: int = 5,
    min_score: float | None = None,
) -> list[Hit]:
    """检索与查询最相关的 chunk。"""
    if not query.strip():
        raise ValueError("查询内容不能为空")
    query_vector = embedder.embed_one(query)
    return store.search(query_vector, top_k=top_k, min_score=min_score)


def format_hits(hits: Sequence[Hit], *, max_chars: int = 300) -> str:
    """把检索结果格式化成便于终端阅读的文本。"""
    if not hits:
        return "（没有命中任何内容）"
    blocks: list[str] = []
    for rank, hit in enumerate(hits, start=1):
        preview = hit.chunk.text.strip().replace("\n", " ")
        if len(preview) > max_chars:
            preview = preview[:max_chars] + "…"
        blocks.append(
            f"[{rank}] score={hit.score:.4f}  {hit.chunk.citation()}\n"
            f"    {preview}"
        )
    return "\n\n".join(blocks)
