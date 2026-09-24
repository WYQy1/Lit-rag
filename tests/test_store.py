"""向量库的单元测试。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lit_rag.schema import Chunk
from lit_rag.store import VectorStore
from lit_rag.store.memory import CHUNKS_FILE, META_FILE, VECTORS_FILE


def make_chunk(index: int, text: str = "") -> Chunk:
    return Chunk(
        chunk_id=f"doc:{index:04d}",
        doc_id="doc",
        text=text or f"chunk number {index}",
        index=index,
        page_start=1,
        page_end=1,
        section="1. Introduction",
        metadata={"source": "demo.pdf", "title": "Demo"},
    )


def unit(*components: float) -> np.ndarray:
    """构造一个 L2 归一化的向量。"""
    vector = np.asarray(components, dtype=np.float32)
    return vector / np.linalg.norm(vector)


class TestAdd:
    def test_size_and_dim(self) -> None:
        store = VectorStore()
        store.add([make_chunk(0), make_chunk(1)], np.eye(2, dtype=np.float32))
        assert store.size == 2
        assert store.dim == 2

    def test_empty_add_is_noop(self) -> None:
        store = VectorStore()
        store.add([], np.zeros((0, 4), dtype=np.float32))
        assert store.is_empty

    def test_count_mismatch_raises(self) -> None:
        store = VectorStore()
        with pytest.raises(ValueError, match="不一致"):
            store.add([make_chunk(0)], np.zeros((2, 4), dtype=np.float32))

    def test_dim_mismatch_on_second_add_raises(self) -> None:
        store = VectorStore()
        store.add([make_chunk(0)], np.zeros((1, 4), dtype=np.float32))
        with pytest.raises(ValueError, match="维度不一致"):
            store.add([make_chunk(1)], np.zeros((1, 8), dtype=np.float32))

    def test_incremental_add_appends(self) -> None:
        store = VectorStore()
        store.add([make_chunk(0)], unit(1.0, 0.0))
        store.add([make_chunk(1)], unit(0.0, 1.0))
        assert store.size == 2
        assert store.dim == 2


class TestSearch:
    def test_returns_most_similar_first(self) -> None:
        store = VectorStore()
        store.add(
            [make_chunk(0, "alpha"), make_chunk(1, "beta"), make_chunk(2, "gamma")],
            np.vstack([unit(1.0, 0.0), unit(0.0, 1.0), unit(0.7, 0.7)]),
        )
        hits = store.search(unit(1.0, 0.0), top_k=3)
        assert hits[0].chunk.text == "alpha"
        assert hits[0].score == pytest.approx(1.0, abs=1e-6)
        assert hits[-1].chunk.text == "beta"

    def test_top_k_limits_results(self) -> None:
        store = VectorStore()
        vectors = np.vstack([unit(1.0, 0.0)] * 5)
        store.add([make_chunk(i) for i in range(5)], vectors)
        assert len(store.search(unit(1.0, 0.0), top_k=2)) == 2

    def test_top_k_larger_than_size_is_clamped(self) -> None:
        store = VectorStore()
        store.add([make_chunk(0)], unit(1.0, 0.0))
        assert len(store.search(unit(1.0, 0.0), top_k=99)) == 1

    def test_min_score_filters(self) -> None:
        store = VectorStore()
        store.add(
            [make_chunk(0), make_chunk(1)],
            np.vstack([unit(1.0, 0.0), unit(0.0, 1.0)]),
        )
        hits = store.search(unit(1.0, 0.0), top_k=5, min_score=0.5)
        assert len(hits) == 1

    def test_empty_store_returns_empty_list(self) -> None:
        assert VectorStore().search(unit(1.0, 0.0)) == []

    def test_invalid_top_k_raises(self) -> None:
        store = VectorStore()
        store.add([make_chunk(0)], unit(1.0, 0.0))
        with pytest.raises(ValueError):
            store.search(unit(1.0, 0.0), top_k=0)

    def test_query_dim_mismatch_raises(self) -> None:
        store = VectorStore()
        store.add([make_chunk(0)], unit(1.0, 0.0))
        with pytest.raises(ValueError, match="维度"):
            store.search(unit(1.0, 0.0, 0.0), top_k=1)


class TestPersistence:
    def test_save_creates_expected_files(self, tmp_path: Path) -> None:
        store = VectorStore()
        store.add([make_chunk(0)], unit(1.0, 0.0))
        store.save(tmp_path)
        assert (tmp_path / VECTORS_FILE).is_file()
        assert (tmp_path / CHUNKS_FILE).is_file()
        assert (tmp_path / META_FILE).is_file()

    def test_roundtrip_preserves_content(self, tmp_path: Path) -> None:
        store = VectorStore()
        store.add(
            [make_chunk(0, "text zero"), make_chunk(1, "text one")],
            np.vstack([unit(1.0, 0.0), unit(0.0, 1.0)]),
        )
        store.save(tmp_path)

        loaded = VectorStore.load(tmp_path)
        assert loaded.size == 2
        assert loaded.dim == 2
        assert [c.text for c in loaded.chunks()] == ["text zero", "text one"]
        assert loaded.search(unit(1.0, 0.0), top_k=1)[0].chunk.text == "text zero"

    def test_embedder_metadata_is_persisted(self, tmp_path: Path) -> None:
        store = VectorStore()
        store.add([make_chunk(0)], unit(1.0, 0.0))
        store.save(tmp_path, extra_meta={"embedder": "offline-hashing"})

        meta = json.loads((tmp_path / META_FILE).read_text(encoding="utf-8"))
        assert meta["embedder"] == "offline-hashing"
        assert meta["size"] == 1
        assert meta["dim"] == 2
        assert VectorStore.load(tmp_path).meta["embedder"] == "offline-hashing"

    def test_saving_empty_store_is_allowed(self, tmp_path: Path) -> None:
        VectorStore(dim=8).save(tmp_path)
        loaded = VectorStore.load(tmp_path)
        assert loaded.is_empty

    def test_load_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            VectorStore.load(tmp_path / "does-not-exist")

    def test_load_detects_corrupted_index(self, tmp_path: Path) -> None:
        store = VectorStore()
        store.add([make_chunk(0), make_chunk(1)], np.vstack([unit(1.0, 0.0), unit(0.0, 1.0)]))
        store.save(tmp_path)

        # 手工删掉一行 chunk，制造不一致
        lines = (tmp_path / CHUNKS_FILE).read_text(encoding="utf-8").splitlines()
        (tmp_path / CHUNKS_FILE).write_text(lines[0] + "\n", encoding="utf-8")

        with pytest.raises(ValueError, match="损坏"):
            VectorStore.load(tmp_path)

    def test_save_creates_nested_directory(self, tmp_path: Path) -> None:
        store = VectorStore()
        store.add([make_chunk(0)], unit(1.0, 0.0))
        target = store.save(tmp_path / "a" / "b")
        assert (target / META_FILE).is_file()

    def test_chunks_are_json_serializable_unicode(self, tmp_path: Path) -> None:
        store = VectorStore()
        store.add([make_chunk(0, "掺杂浓度的影响")], unit(1.0, 0.0))
        store.save(tmp_path)
        raw = (tmp_path / CHUNKS_FILE).read_text(encoding="utf-8")
        assert "掺杂浓度的影响" in raw  # ensure_ascii=False，中文不该被转义
