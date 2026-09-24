"""文本清洗模块的单元测试。"""

from __future__ import annotations

from lit_rag.parsing.cleaner import (
    clean_pages,
    dehyphenate,
    detect_repeated_lines,
    normalize_whitespace,
    split_references,
    strip_page_numbers,
)


class TestDehyphenate:
    def test_joins_word_broken_across_lines(self) -> None:
        assert dehyphenate("this is an analy-\nsis of the data") == "this is an analysis of the data"

    def test_handles_trailing_space_before_newline(self) -> None:
        assert dehyphenate("spec-\n  trum") == "spectrum"

    def test_keeps_hyphen_when_next_line_is_uppercase(self) -> None:
        # "Cr3+-\nDoped" 里的连字符是内容的一部分，不该被合并
        assert dehyphenate("Cr3+-\nDoped garnet") == "Cr3+-\nDoped garnet"

    def test_keeps_normal_hyphenated_word(self) -> None:
        assert dehyphenate("well-known fact") == "well-known fact"


class TestNormalizeWhitespace:
    def test_collapses_runs_of_spaces(self) -> None:
        assert normalize_whitespace("a    b\t\tc") == "a b c"

    def test_collapses_excess_blank_lines(self) -> None:
        assert normalize_whitespace("a\n\n\n\n\nb") == "a\n\nb"

    def test_preserves_paragraph_break(self) -> None:
        assert normalize_whitespace("first\n\nsecond") == "first\n\nsecond"

    def test_normalizes_line_endings(self) -> None:
        assert normalize_whitespace("a\r\nb\rc") == "a\nb\nc"

    def test_replaces_non_breaking_spaces(self) -> None:
        assert normalize_whitespace("a\u00a0b") == "a b"


class TestStripPageNumbers:
    def test_removes_bare_number_line(self) -> None:
        assert strip_page_numbers("text\n42\nmore text") == "text\nmore text"

    def test_removes_page_x_of_y(self) -> None:
        assert strip_page_numbers("text\nPage 3 of 12\nmore") == "text\nmore"

    def test_removes_roman_numeral_page(self) -> None:
        assert strip_page_numbers("text\niv\nmore") == "text\nmore"

    def test_keeps_numbers_inside_sentences(self) -> None:
        text = "The sample was heated to 1200 C for 4 h."
        assert strip_page_numbers(text) == text

    def test_keeps_year_like_line(self) -> None:
        # "2021" 单独成行时确实像页码，但这里只验证不会误删带文字的年份行
        text = "Published in 2021 by Demo Press"
        assert strip_page_numbers(text) == text


class TestDetectRepeatedLines:
    def test_detects_running_header(self) -> None:
        pages = [
            "J. Lumin. Demo\nBody of page one.\n1",
            "J. Lumin. Demo\nBody of page two.\n2",
            "J. Lumin. Demo\nBody of page three.\n3",
        ]
        repeated = detect_repeated_lines(pages)
        assert "J. Lumin. Demo" in repeated

    def test_single_page_has_no_repeats(self) -> None:
        assert detect_repeated_lines(["only page"]) == set()

    def test_body_text_is_not_marked_repeated(self) -> None:
        pages = [
            "Header\nUnique sentence about garnet.\n1",
            "Header\nAnother unique sentence about garnet.\n2",
        ]
        repeated = detect_repeated_lines(pages)
        assert "Unique sentence about garnet." not in repeated


class TestCleanPages:
    def test_removes_header_and_page_numbers_but_keeps_body(self) -> None:
        pages = [
            "Journal of Demo\nFirst page body text.\n1",
            "Journal of Demo\nSecond page body text.\n2",
        ]
        cleaned = clean_pages(pages)
        joined = "\n".join(cleaned)
        assert "Journal of Demo" not in joined
        assert "First page body text." in joined
        assert "Second page body text." in joined
        assert "\n1" not in joined

    def test_returns_same_number_of_pages(self) -> None:
        pages = ["a", "b", "c"]
        assert len(clean_pages(pages)) == len(pages)

    def test_drop_references_truncates(self) -> None:
        pages = ["Body text here.\nReferences\n[1] Someone et al. 2021."]
        cleaned = clean_pages(pages, drop_references=True)
        assert "Body text here." in cleaned[0]
        assert "[1] Someone" not in cleaned[0]


class TestSplitReferences:
    def test_splits_at_references_heading(self) -> None:
        body, refs = split_references("Body paragraph.\nReferences\n[1] A. B. 2020.")
        assert body == "Body paragraph."
        assert refs.startswith("References")

    def test_returns_original_when_no_heading(self) -> None:
        text = "Just body text without any bibliography section."
        body, refs = split_references(text)
        assert body == text
        assert refs == ""

    def test_recognizes_numbered_heading(self) -> None:
        body, refs = split_references("Body.\n5. References\n[1] X.")
        assert body == "Body."
        assert refs != ""
