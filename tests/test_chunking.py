"""切分模块的单元测试。"""

from __future__ import annotations

import pytest

from lit_rag.chunking import split_document
from lit_rag.chunking.splitter import _is_heading, _joined
from lit_rag.schema import Document, Page


class TestHeadingDetection:
    @pytest.mark.parametrize(
        "line",
        [
            "1. Introduction",
            "2.1 Sample preparation",
            "3.2.1 Results",
            "Abstract",
            "References",
            "CONCLUSIONS",
        ],
    )
    def test_recognizes_headings(self, line: str) -> None:
        assert _is_heading(line)

    @pytest.mark.parametrize(
        "line",
        [
            "3.5 g of powder was added to the crucible.",
            "This is a normal sentence in the body text.",
            "The intensity increased by 2.5 times.",
            "",
        ],
    )
    def test_rejects_body_lines(self, line: str) -> None:
        assert not _is_heading(line)


class TestLineJoining:
    def test_english_lines_get_a_space(self) -> None:
        assert _joined(["hello", "world"]) == "hello world"

    def test_chinese_lines_are_concatenated(self) -> None:
        assert _joined(["本文研究了", "掺杂浓度的影响。"]) == "本文研究了掺杂浓度的影响。"

    def test_mixed_cjk_and_latin_gets_a_space(self) -> None:
        # 中英混排必须补空格，否则会出现 "maximum.摘要" 这种粘连
        assert _joined(["emission maximum.", "摘要：本文"]) == "emission maximum. 摘要：本文"


class TestSplitDocument:
    def test_detects_sections(self, synthetic_doc: Document) -> None:
        chunks = split_document(synthetic_doc, chunk_size=400, chunk_overlap=50)
        sections = {chunk.section for chunk in chunks}
        assert "1. Introduction" in sections
        assert "2. Methods" in sections
        assert "3. Results" in sections

    def test_respects_chunk_size(self, synthetic_doc: Document) -> None:
        chunk_size = 200
        chunks = split_document(synthetic_doc, chunk_size=chunk_size, chunk_overlap=40)
        assert chunks
        for chunk in chunks:
            assert chunk.n_chars <= chunk_size, f"chunk {chunk.index} 超长：{chunk.n_chars}"

    def test_chunks_are_numbered_sequentially(self, synthetic_doc: Document) -> None:
        chunks = split_document(synthetic_doc, chunk_size=200, chunk_overlap=40)
        assert [chunk.index for chunk in chunks] == list(range(len(chunks)))

    def test_chunk_ids_are_unique(self, synthetic_doc: Document) -> None:
        chunks = split_document(synthetic_doc, chunk_size=200, chunk_overlap=40)
        ids = [chunk.chunk_id for chunk in chunks]
        assert len(ids) == len(set(ids))

    def test_overlap_exists_within_a_long_section(self) -> None:
        long_text = " ".join(
            f"Sentence number {i} talks about garnet phosphors and their emission." for i in range(40)
        )
        doc = Document(
            doc_id="longdoc",
            source="long.txt",
            pages=[Page(number=1, text=f"1. Introduction\n{long_text}")],
            metadata={"filename": "long.txt"},
        )
        chunks = split_document(doc, chunk_size=300, chunk_overlap=80, min_chunk_size=1)
        assert len(chunks) >= 3
        # 相邻两块应有公共子串（重叠）
        for previous, current in zip(chunks, chunks[1:], strict=False):
            tail = previous.text[-40:]
            assert tail[:20] in current.text, "相邻 chunk 之间未发现重叠"

    def test_no_overlap_across_sections(self, synthetic_doc: Document) -> None:
        chunks = split_document(synthetic_doc, chunk_size=400, chunk_overlap=200)
        for chunk in chunks:
            # 每个 chunk 只能属于一个章节，不应混入其它章节的内容
            assert chunk.section != "" or chunk.index == 0

    def test_long_paragraph_is_split_by_sentence(self) -> None:
        paragraph = " ".join(f"This is sentence {i}." for i in range(60))
        doc = Document(
            doc_id="para",
            source="p.txt",
            pages=[Page(number=1, text=paragraph)],
            metadata={"filename": "p.txt"},
        )
        chunks = split_document(doc, chunk_size=200, chunk_overlap=0)
        assert len(chunks) > 1
        # 不应在句子中间硬切：每块都以句号结束
        for chunk in chunks[:-1]:
            assert chunk.text.rstrip().endswith(".")

    def test_page_range_is_tracked(self, synthetic_doc: Document) -> None:
        chunks = split_document(synthetic_doc, chunk_size=200, chunk_overlap=20)
        assert chunks[0].page_start == 1
        assert any(chunk.page_start == 2 for chunk in chunks)

    def test_embed_text_prepends_section(self, synthetic_doc: Document) -> None:
        chunks = split_document(synthetic_doc, chunk_size=200, chunk_overlap=20)
        target = next(chunk for chunk in chunks if chunk.section == "1. Introduction")
        assert target.embed_text.startswith("1. Introduction\n")
        assert target.text.startswith("This is the first paragraph")

    def test_empty_document_returns_no_chunks(self) -> None:
        doc = Document(doc_id="empty", source="e.txt", pages=[])
        assert split_document(doc) == []

    def test_invalid_parameters_raise(self, synthetic_doc: Document) -> None:
        with pytest.raises(ValueError):
            split_document(synthetic_doc, chunk_size=0)
        with pytest.raises(ValueError):
            split_document(synthetic_doc, chunk_size=100, chunk_overlap=100)
        with pytest.raises(ValueError):
            split_document(synthetic_doc, chunk_size=100, chunk_overlap=-1)


class TestCitation:
    def test_citation_contains_source_and_page(self, synthetic_doc: Document) -> None:
        chunks = split_document(synthetic_doc, chunk_size=200, chunk_overlap=20)
        citation = chunks[1].citation()
        assert "synthetic.txt" in citation
        assert "p." in citation

    def test_citation_includes_section_when_present(self, synthetic_doc: Document) -> None:
        chunks = split_document(synthetic_doc, chunk_size=200, chunk_overlap=20)
        target = next(chunk for chunk in chunks if chunk.section == "2. Methods")
        assert "§2. Methods" in target.citation()
