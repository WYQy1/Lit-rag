"""PDF 解析与文本清洗。

解析默认优先使用 PyMuPDF（对多栏排版、公式、表格的文本抽取质量更好），
若不可用则自动回退到纯 Python 的 pypdf。
"""

from __future__ import annotations

from lit_rag.parsing.cleaner import (
    clean_pages,
    dehyphenate,
    detect_repeated_lines,
    normalize_whitespace,
    split_references,
    strip_page_numbers,
)
from lit_rag.parsing.loader import (
    PARSING_AVAILABLE,
    iter_pdf_paths,
    load_pdf,
    load_pdfs,
)

__all__ = [
    "PARSING_AVAILABLE",
    "clean_pages",
    "dehyphenate",
    "detect_repeated_lines",
    "iter_pdf_paths",
    "load_pdf",
    "load_pdfs",
    "normalize_whitespace",
    "split_references",
    "strip_page_numbers",
]
