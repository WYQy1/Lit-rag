"""pytest 共享夹具。

约定：所有测试都必须在**无网络、无 API Key** 的环境下通过，
因此默认只使用离线 embedder 与仓库内置的样本 PDF。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lit_rag.schema import Document, Page

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = PROJECT_ROOT / "samples"
SAMPLE_PDF = SAMPLES_DIR / "nir-phosphor-demo.pdf"


@pytest.fixture(scope="session")
def sample_pdf() -> Path:
    """仓库内置的样本 PDF（由 scripts/make_sample_pdf.py 生成）。"""
    if not SAMPLE_PDF.is_file():
        pytest.skip(f"样本 PDF 不存在，请先运行 scripts/make_sample_pdf.py：{SAMPLE_PDF}")
    return SAMPLE_PDF


@pytest.fixture
def synthetic_doc() -> Document:
    """构造一份带章节、空行、中英混排的文档，用于切分逻辑测试。"""
    page_one = "\n".join(
        [
            "A Synthetic Paper on Phosphors",
            "1. Introduction",
            "This is the first paragraph of the introduction. It has two sentences.",
            "This line continues the very same paragraph across a physical line break.",
            "",
            "This paragraph follows a blank line and should start a new block.",
            "2. Methods",
            "We used a standard solid state procedure for all samples.",
        ]
    )
    page_two = "\n".join(
        [
            "摘要：本文研究了掺杂浓度对发光性能的影响。",
            "3. Results",
            "The emission maximum shifted to longer wavelength as doping increased.",
        ]
    )
    return Document(
        doc_id="synthetic0001",
        source="synthetic.txt",
        title="A Synthetic Paper on Phosphors",
        pages=[Page(number=1, text=page_one), Page(number=2, text=page_two)],
        metadata={"filename": "synthetic.txt", "n_pages": 2},
    )
