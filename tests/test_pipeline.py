"""端到端流水线测试：解析 → 切分 → 向量化 → 入库 → 检索。

全程使用离线 embedder 与仓库内置样本 PDF，无网络、无 API Key。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lit_rag.config import Settings
from lit_rag.embedding import HashingEmbedder
from lit_rag.pipeline import (
    chunks_from_documents,
    format_hits,
    index_chunks,
    ingest,
    load_documents,
    resolve_pdf_paths,
    retrieve,
)
from lit_rag.store import VectorStore


@pytest.fixture
def settings() -> Settings:
    return Settings(chunk_size=500, chunk_overlap=80, min_chunk_size=50)


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=1024)


class TestResolvePdfPaths:
    def test_expands_directory(self, sample_pdf: Path) -> None:
        assert sample_pdf in resolve_pdf_paths([sample_pdf.parent])

    def test_deduplicates_overlapping_inputs(self, sample_pdf: Path) -> None:
        paths = resolve_pdf_paths([sample_pdf, sample_pdf, sample_pdf.parent])
        assert paths == [sample_pdf]

    def test_ignores_missing_paths(self, tmp_path: Path) -> None:
        assert resolve_pdf_paths([tmp_path / "nope.pdf"]) == []

    def test_keeps_single_file(self, sample_pdf: Path) -> None:
        assert resolve_pdf_paths([sample_pdf]) == [sample_pdf]


class TestIngest:
    def test_builds_non_empty_index(self, sample_pdf: Path, settings: Settings) -> None:
        store = ingest([sample_pdf], settings=settings, embedder=HashingEmbedder(dim=512))
        assert store.size > 0
        assert store.dim == 512

    def test_raises_when_no_pdf_found(self, tmp_path: Path, settings: Settings) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            ingest([empty_dir], settings=settings, embedder=HashingEmbedder(dim=64))

    def test_uses_settings_for_chunking(self, sample_pdf: Path) -> None:
        coarse = Settings(chunk_size=1500, chunk_overlap=100)
        fine = Settings(chunk_size=300, chunk_overlap=50)

        coarse_store = ingest([sample_pdf], settings=coarse, embedder=HashingEmbedder(dim=256))
        fine_store = ingest([sample_pdf], settings=fine, embedder=HashingEmbedder(dim=256))
        assert fine_store.size > coarse_store.size


class TestRetrieve:
    def test_returns_hits_with_positive_scores(
        self, sample_pdf: Path, settings: Settings, embedder: HashingEmbedder
    ) -> None:
        store = ingest([sample_pdf], settings=settings, embedder=embedder)
        hits = retrieve("near infrared emission of garnet phosphors", store, embedder, top_k=3)
        assert hits
        assert all(hit.score > 0 for hit in hits)

    def test_finds_the_relevant_section(
        self, sample_pdf: Path, settings: Settings, embedder: HashingEmbedder
    ) -> None:
        store = ingest([sample_pdf], settings=settings, embedder=embedder)
        hits = retrieve(
            "concentration quenching and lifetime shortening", store, embedder, top_k=3
        )
        sections = [hit.chunk.section for hit in hits]
        assert any("Luminescence" in section for section in sections), sections

    def test_citation_points_back_to_source(
        self, sample_pdf: Path, settings: Settings, embedder: HashingEmbedder
    ) -> None:
        store = ingest([sample_pdf], settings=settings, embedder=embedder)
        hit = retrieve("garnet crystal structure", store, embedder, top_k=1)[0]
        assert sample_pdf.name in hit.chunk.citation()

    def test_blank_query_raises(
        self, sample_pdf: Path, settings: Settings, embedder: HashingEmbedder
    ) -> None:
        store = ingest([sample_pdf], settings=settings, embedder=embedder)
        with pytest.raises(ValueError):
            retrieve("   ", store, embedder)

    def test_results_are_sorted_by_score(
        self, sample_pdf: Path, settings: Settings, embedder: HashingEmbedder
    ) -> None:
        store = ingest([sample_pdf], settings=settings, embedder=embedder)
        hits = retrieve("emission", store, embedder, top_k=5)
        scores = [hit.score for hit in hits]
        assert scores == sorted(scores, reverse=True)


class TestPersistenceRoundTrip:
    def test_save_then_load_then_retrieve(
        self,
        sample_pdf: Path,
        settings: Settings,
        embedder: HashingEmbedder,
        tmp_path: Path,
    ) -> None:
        store = ingest([sample_pdf], settings=settings, embedder=embedder)
        store.save(tmp_path, extra_meta={"embedder": embedder.name})

        loaded = VectorStore.load(tmp_path)
        assert loaded.size == store.size
        assert loaded.meta["embedder"] == embedder.name

        hits = retrieve("broadband near infrared", loaded, embedder, top_k=2)
        assert hits


class TestHelpers:
    def test_index_chunks_matches_chunk_count(self, sample_pdf: Path, settings: Settings) -> None:
        docs = load_documents([sample_pdf], settings=settings)
        chunks = chunks_from_documents(docs, settings=settings)
        store = index_chunks(chunks, HashingEmbedder(dim=256))
        assert store.size == len(chunks)

    def test_index_empty_chunks_gives_empty_store(self) -> None:
        assert index_chunks([], HashingEmbedder(dim=64)).is_empty

    def test_format_hits_handles_no_results(self) -> None:
        assert "没有命中" in format_hits([])

    def test_format_hits_shows_score_and_source(
        self, sample_pdf: Path, settings: Settings, embedder: HashingEmbedder
    ) -> None:
        store = ingest([sample_pdf], settings=settings, embedder=embedder)
        hits = retrieve("emission", store, embedder, top_k=1)
        rendered = format_hits(hits)
        assert "score=" in rendered
        assert sample_pdf.name in rendered

    def test_format_hits_truncates_long_text(self) -> None:
        from lit_rag.schema import Chunk, Hit

        chunk = Chunk(
            chunk_id="c1",
            doc_id="d1",
            text="x" * 1000,
            index=0,
            metadata={"source": "s.pdf"},
        )
        rendered = format_hits([Hit(chunk=chunk, score=0.5)], max_chars=50)
        assert "…" in rendered
        assert len(rendered) < 300
