"""PDF 文件加载：把磁盘上的 PDF 变成 :class:`~lit_rag.schema.Document`。

两个解析后端：

- **PyMuPDF**（``pymupdf``）—— 默认优先，文本抽取质量更好，速度快
- **pypdf** —— 纯 Python，作为兜底，任何环境都能装

``doc_id`` 使用文件内容的 SHA-1 前 16 位，因此同一份文件重复入库会得到
相同的 ID，天然支持去重与增量更新。
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from lit_rag.parsing.cleaner import clean_pages
from lit_rag.schema import Document, Page

logger = logging.getLogger(__name__)

_HASH_BLOCK = 1 << 20  # 1 MiB


def _import_pymupdf():
    """延迟导入 PyMuPDF，兼容新旧包名（``pymupdf`` / ``fitz``）。"""
    try:
        import pymupdf  # type: ignore[import-not-found]

        return pymupdf
    except ImportError:
        pass
    try:
        import fitz  # type: ignore[import-not-found]

        return fitz
    except ImportError:
        return None


def _import_pypdf():
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]

        return PdfReader
    except ImportError:
        return None


PARSING_AVAILABLE = _import_pymupdf() is not None or _import_pypdf() is not None


def file_digest(path: Path) -> str:
    """计算文件内容的 SHA-1（取前 16 位十六进制）。"""
    digest = hashlib.sha1()  # noqa: S324 - 仅用于生成稳定 ID，非安全用途
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_HASH_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def _extract_with_pymupdf(path: Path) -> tuple[list[str], dict[str, object]]:
    module = _import_pymupdf()
    if module is None:  # pragma: no cover - 由调用方保证
        raise RuntimeError("PyMuPDF 不可用")
    doc = module.open(str(path))
    try:
        pages = [page.get_text("text") or "" for page in doc]
        metadata: dict[str, object] = dict(doc.metadata or {})
    finally:
        doc.close()
    return pages, metadata


def _extract_with_pypdf(path: Path) -> tuple[list[str], dict[str, object]]:
    reader_cls = _import_pypdf()
    if reader_cls is None:  # pragma: no cover - 由调用方保证
        raise RuntimeError("pypdf 不可用")
    reader = reader_cls(str(path))
    if getattr(reader, "is_encrypted", False):
        try:
            reader.decrypt("")
        except Exception as exc:  # noqa: BLE001 - 加密 PDF 无法解析，直接报错更清晰
            raise ValueError(f"PDF 已加密且无法用空密码解密：{path.name}") from exc
    pages = [page.extract_text() or "" for page in reader.pages]
    raw_meta = getattr(reader, "metadata", None) or {}
    metadata: dict[str, object] = {str(k): str(v) for k, v in dict(raw_meta).items()}
    return pages, metadata


def _pick_backend(backend: str) -> str:
    if backend == "auto":
        return "pymupdf" if _import_pymupdf() is not None else "pypdf"
    if backend == "pymupdf" and _import_pymupdf() is None:
        raise RuntimeError("指定了 pymupdf 后端，但未安装 PyMuPDF：pip install pymupdf")
    if backend == "pypdf" and _import_pypdf() is None:
        raise RuntimeError("指定了 pypdf 后端，但未安装 pypdf：pip install pypdf")
    return backend


def load_pdf(
    path: str | Path,
    *,
    backend: str = "auto",
    clean: bool = True,
    drop_references: bool = False,
) -> Document:
    """解析单个 PDF。

    Args:
        path: PDF 文件路径。
        backend: ``auto`` / ``pymupdf`` / ``pypdf``。
        clean: 是否执行文本清洗（去页眉页脚、修断词、删页码）。
        drop_references: 清洗时是否裁掉参考文献列表。

    Returns:
        解析好的 :class:`Document`。
    """
    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"找不到 PDF 文件：{pdf_path}")

    chosen = _pick_backend(backend)
    if chosen == "pymupdf":
        raw_pages, metadata = _extract_with_pymupdf(pdf_path)
    else:
        raw_pages, metadata = _extract_with_pypdf(pdf_path)

    pages_text = (
        clean_pages(raw_pages, drop_references=drop_references) if clean else raw_pages
    )

    title = str(metadata.get("title") or "").strip()
    if not title:
        title = _guess_title(pages_text)

    return Document(
        doc_id=file_digest(pdf_path),
        source=str(pdf_path),
        title=title,
        pages=[Page(number=i, text=text) for i, text in enumerate(pages_text, start=1)],
        metadata={
            "filename": pdf_path.name,
            "backend": chosen,
            "n_pages": len(pages_text),
            **{k: v for k, v in metadata.items() if k in {"author", "subject", "creationDate"}},
        },
    )


def _guess_title(pages: list[str], *, max_len: int = 200) -> str:
    """从第一页里猜标题：取第一个长度合适、且不是页眉噪声的行。"""
    if not pages:
        return ""
    for line in pages[0].split("\n"):
        candidate = line.strip()
        if 10 <= len(candidate) <= max_len and not candidate[0].isdigit():
            return candidate
    return ""


def load_pdfs(
    paths: list[str | Path],
    *,
    backend: str = "auto",
    clean: bool = True,
    drop_references: bool = False,
    on_error: str = "warn",
) -> list[Document]:
    """批量解析 PDF。

    Args:
        paths: 文件路径列表。
        backend: 解析后端。
        clean: 是否清洗。
        drop_references: 是否裁掉参考文献。
        on_error: ``warn`` 跳过失败文件，``raise`` 直接抛出。

    Returns:
        成功解析的 :class:`Document` 列表。
    """
    documents: list[Document] = []
    for item in paths:
        try:
            documents.append(
                load_pdf(
                    item,
                    backend=backend,
                    clean=clean,
                    drop_references=drop_references,
                )
            )
        except Exception as exc:  # noqa: BLE001 - 批量场景下按策略决定是否中断
            if on_error == "raise":
                raise
            logger.warning("解析失败，已跳过 %s：%s", item, exc)
    return documents


def iter_pdf_paths(root: str | Path, *, recursive: bool = True) -> list[Path]:
    """列出目录下所有 PDF（按路径排序，保证结果稳定）。"""
    base = Path(root)
    if base.is_file():
        return [base] if base.suffix.lower() == ".pdf" else []
    if not base.is_dir():
        raise NotADirectoryError(f"既不是文件也不是目录：{base}")
    pattern = "**/*.pdf" if recursive else "*.pdf"
    return sorted(p for p in base.glob(pattern) if p.is_file())
