"""向量化模块的单元测试（全部离线，不依赖任何 API Key）。"""

from __future__ import annotations

import numpy as np
import pytest

from lit_rag.config import Settings
from lit_rag.embedding import HashingEmbedder, create_embedder


class TestHashingEmbedder:
    def test_reports_configured_dim(self) -> None:
        assert HashingEmbedder(dim=256).dim == 256

    def test_output_shape(self) -> None:
        embedder = HashingEmbedder(dim=128)
        matrix = embedder.embed(["one", "two", "three"])
        assert matrix.shape == (3, 128)
        assert matrix.dtype == np.float32

    def test_empty_input_returns_empty_matrix(self) -> None:
        matrix = HashingEmbedder(dim=64).embed([])
        assert matrix.shape == (0, 64)

    def test_rows_are_l2_normalized(self) -> None:
        matrix = HashingEmbedder(dim=256).embed(["hello world", "你好世界"])
        norms = np.linalg.norm(matrix, axis=1)
        np.testing.assert_allclose(norms, np.ones(2), atol=1e-5)

    def test_blank_text_does_not_produce_nan(self) -> None:
        matrix = HashingEmbedder(dim=64).embed(["", "   "])
        assert not np.isnan(matrix).any()

    def test_deterministic_across_instances_and_processes(self) -> None:
        """哈希必须走 blake2b 而不是内置 hash()，否则跨进程不可复现。"""
        first = HashingEmbedder(dim=512).embed(["garnet phosphor"])
        second = HashingEmbedder(dim=512).embed(["garnet phosphor"])
        np.testing.assert_allclose(first, second, rtol=0, atol=0)

    def test_identical_text_scores_one(self) -> None:
        embedder = HashingEmbedder(dim=512)
        vector = embedder.embed(["Cr3+ doped garnet"])[0]
        assert float(vector @ vector) == pytest.approx(1.0, abs=1e-5)

    def test_related_text_scores_higher_than_unrelated(self) -> None:
        embedder = HashingEmbedder(dim=1024)
        vectors = embedder.embed(
            [
                "chromium doping concentration in garnet phosphors",
                "chromium doping level of garnet phosphors",
                "a recipe for tomato pasta with basil",
            ]
        )
        related = float(vectors[0] @ vectors[1])
        unrelated = float(vectors[0] @ vectors[2])
        assert related > unrelated

    def test_handles_chinese_text(self) -> None:
        embedder = HashingEmbedder(dim=512)
        vectors = embedder.embed(["掺杂浓度对发光性能的影响", "掺杂浓度对发光性能的影响"])
        assert float(vectors[0] @ vectors[1]) == pytest.approx(1.0, abs=1e-5)

    def test_embed_one_returns_1d_vector(self) -> None:
        vector = HashingEmbedder(dim=128).embed_one("single")
        assert vector.shape == (128,)

    def test_invalid_dim_raises(self) -> None:
        with pytest.raises(ValueError):
            HashingEmbedder(dim=0)

    def test_invalid_ngram_range_raises(self) -> None:
        with pytest.raises(ValueError):
            HashingEmbedder(ngram_range=(4, 2))
        with pytest.raises(ValueError):
            HashingEmbedder(ngram_range=(0, 3))


class TestCreateEmbedder:
    def test_offline_provider(self) -> None:
        settings = Settings(embed_provider="offline")
        embedder = create_embedder(settings)
        assert isinstance(embedder, HashingEmbedder)

    def test_openai_provider_without_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in (
            "EMBED_API_KEY",
            "SILICONFLOW_API_KEY",
            "ZHIPUAI_API_KEY",
            "DASHSCOPE_API_KEY",
            "OPENAI_API_KEY",
            "LIT_RAG_EMBED_API_KEY",
        ):
            monkeypatch.delenv(name, raising=False)

        settings = Settings(embed_provider="openai", embed_api_key=None)
        with pytest.raises(ValueError, match="API Key"):
            create_embedder(settings)

    def test_openai_provider_with_explicit_key(self) -> None:
        settings = Settings(
            embed_provider="openai",
            embed_model="BAAI/bge-m3",
            embed_api_key="sk-test-not-a-real-key",
            embed_base_url="https://example.invalid/v1",
        )
        embedder = create_embedder(settings)
        assert embedder.name == "openai-compat(BAAI/bge-m3)"

    def test_openai_embedder_rejects_empty_input_before_dim_known(self) -> None:
        """维度未知时调用 embed([]) 应给出明确报错，而不是静默返回错误形状。"""
        settings = Settings(embed_provider="openai", embed_api_key="sk-test")
        embedder = create_embedder(settings)
        with pytest.raises(RuntimeError, match="维度"):
            embedder.embed([])

    def test_missing_api_key_error_mentions_deepseek_limitation(self) -> None:
        settings = Settings(embed_provider="openai", embed_api_key=None)
        try:
            create_embedder(settings)
        except ValueError as exc:
            assert "DeepSeek" in str(exc) or "embeddings" in str(exc)
        else:  # pragma: no cover - 环境里存在 fallback key 时才会走到
            pytest.skip("环境中存在可用的 fallback API Key")
