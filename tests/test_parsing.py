"""PDF 解析模块测试。

使用仓库内置的样本 PDF（``samples/nir-phosphor-demo.pdf``），
因此**不需要网络、不需要 API Key**，在 CI 中可完整运行。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lit_rag.parsing import iter_pdf_paths, load_pdf, load_pdfs


class TestLoadPdf:
    def test_parses_sample_document(self, sample_pdf: Path) -> None:
        doc = load_pdf(sample_pdf)
        assert doc.n_pages >= 2
        assert doc.n_chars > 1000
        assert doc.metadata["filename"] == sample_pdf.name

    def test_doc_id_is_content_addressed_and_stable(self, sample_pdf: Path) -> None:
        first = load_pdf(sample_pdf)
        second = load_pdf(sample_pdf)
        assert first.doc_id == second.doc_id
        assert len(first.doc_id) == 16

    def test_title_comes_from_pdf_metadata(self, sample_pdf: Path) -> None:
        title = load_pdf(sample_pdf).title
        assert "Garnet" in title or "Cr3+" in title

    def test_running_header_is_removed(self, sample_pdf: Path) -> None:
        # 样本每页都印了同一行页眉，跨页重复检测应把它清掉
        assert "Submitted Manuscript" not in load_pdf(sample_pdf).text

    def test_bare_page_number_lines_are_removed(self, sample_pdf: Path) -> None:
        leftovers = [
            line for line in load_pdf(sample_pdf).text.split("\n") if line.strip().isdigit()
        ]
        assert leftovers == []

    def test_all_section_headings_survive_cleaning(self, sample_pdf: Path) -> None:
        text = load_pdf(sample_pdf).text
        for heading in (
            "Abstract",
            "1. Introduction",
            "2. Experimental",
            "3. Results and Discussion",
            "4. Conclusions",
            "References",
        ):
            assert heading in text, f"章节标题丢失：{heading}"

    def test_chinese_abstract_is_extracted(self, sample_pdf: Path) -> None:
        assert "掺杂浓度" in load_pdf(sample_pdf).text

    def test_clean_false_keeps_raw_noise(self, sample_pdf: Path) -> None:
        raw = load_pdf(sample_pdf, clean=False)
        assert "Submitted Manuscript" in raw.text

    def test_drop_references_removes_bibliography(self, sample_pdf: Path) -> None:
        doc = load_pdf(sample_pdf, drop_references=True)
        assert "[1] A. Demo" not in doc.text
        assert "4. Conclusions" in doc.text

    def test_both_backends_extract_text(self, sample_pdf: Path) -> None:
        """PyMuPDF 与 pypdf 都应能抽出正文，方便在缺依赖时兜底。"""
        fast = load_pdf(sample_pdf, backend="pymupdf")
        fallback = load_pdf(sample_pdf, backend="pypdf")
        assert fast.n_chars > 0
        assert fallback.n_chars > 0
        assert "garnet" in fast.text.lower()
        assert "garnet" in fallback.text.lower()

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_pdf(tmp_path / "not-here.pdf")

    def test_page_numbers_are_one_based(self, sample_pdf: Path) -> None:
        doc = load_pdf(sample_pdf)
        assert [page.number for page in doc.pages] == list(range(1, doc.n_pages + 1))


class TestLoadPdfs:
    def test_loads_multiple_files(self, sample_pdf: Path) -> None:
        docs = load_pdfs([sample_pdf, sample_pdf])
        assert len(docs) == 2

    def test_on_error_warn_skips_bad_file(self, tmp_path: Path) -> None:
        docs = load_pdfs([tmp_path / "missing.pdf"], on_error="warn")
        assert docs == []

    def test_on_error_raise_propagates(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_pdfs([tmp_path / "missing.pdf"], on_error="raise")


class TestIterPdfPaths:
    def test_finds_pdfs_in_directory(self, sample_pdf: Path) -> None:
        assert sample_pdf in iter_pdf_paths(sample_pdf.parent)

    def test_single_file_is_returned_as_is(self, sample_pdf: Path) -> None:
        assert iter_pdf_paths(sample_pdf) == [sample_pdf]

    def test_non_pdf_file_returns_empty(self, tmp_path: Path) -> None:
        txt = tmp_path / "notes.txt"
        txt.write_text("hello", encoding="utf-8")
        assert iter_pdf_paths(txt) == []

    def test_missing_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(NotADirectoryError):
            iter_pdf_paths(tmp_path / "nowhere")

    def test_non_recursive_mode(self, sample_pdf: Path) -> None:
        paths = iter_pdf_paths(sample_pdf.parent, recursive=False)
        assert sample_pdf in paths

    def test_results_are_sorted(self, sample_pdf: Path) -> None:
        paths = iter_pdf_paths(sample_pdf.parent)
        assert paths == sorted(paths)
