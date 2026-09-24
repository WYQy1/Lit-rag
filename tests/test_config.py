"""配置模块的单元测试。"""

from __future__ import annotations

from dataclasses import replace

import pytest

from lit_rag.config import Settings


class TestValidation:
    def test_defaults_are_valid(self) -> None:
        settings = Settings()
        assert settings.chunk_size == 800
        assert settings.chunk_overlap < settings.chunk_size
        assert settings.embed_provider == "offline"

    def test_overlap_must_be_smaller_than_chunk_size(self) -> None:
        with pytest.raises(ValueError, match="必须小于"):
            Settings(chunk_size=100, chunk_overlap=100)

    def test_negative_overlap_raises(self) -> None:
        with pytest.raises(ValueError):
            Settings(chunk_overlap=-1)

    def test_zero_chunk_size_raises(self) -> None:
        with pytest.raises(ValueError):
            Settings(chunk_size=0)

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="embed_provider"):
            Settings(embed_provider="magic")

    def test_unknown_pdf_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="pdf_backend"):
            Settings(pdf_backend="ghost")


class TestFromEnv:
    def test_reads_prefixed_variables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LIT_RAG_CHUNK_SIZE", "500")
        monkeypatch.setenv("LIT_RAG_CHUNK_OVERLAP", "50")
        monkeypatch.setenv("LIT_RAG_EMBED_PROVIDER", "offline")
        monkeypatch.setenv("LIT_RAG_TOP_K", "3")

        settings = Settings.from_env()
        assert settings.chunk_size == 500
        assert settings.chunk_overlap == 50
        assert settings.top_k == 3

    def test_falls_back_to_defaults_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in ("CHUNK_SIZE", "CHUNK_OVERLAP", "EMBED_PROVIDER", "TOP_K"):
            monkeypatch.delenv(f"LIT_RAG_{name}", raising=False)
        settings = Settings.from_env()
        assert settings.chunk_size == 800

    def test_non_integer_value_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LIT_RAG_CHUNK_SIZE", "abc")
        with pytest.raises(ValueError, match="整数"):
            Settings.from_env()

    def test_boolean_flag_parsing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LIT_RAG_DROP_REFERENCES", "true")
        assert Settings.from_env().drop_references is True
        monkeypatch.setenv("LIT_RAG_DROP_REFERENCES", "no")
        assert Settings.from_env().drop_references is False


class TestDescribe:
    def test_masks_api_key(self) -> None:
        settings = Settings(embed_api_key="sk-abcdefghijklmnop")
        described = settings.describe()
        assert described["embed_api_key"] != "sk-abcdefghijklmnop"
        assert "..." in str(described["embed_api_key"])

    def test_short_key_is_fully_masked(self) -> None:
        settings = Settings(embed_api_key="short")
        assert settings.describe()["embed_api_key"] == "***"

    def test_missing_key_stays_none(self) -> None:
        settings = replace(Settings(), embed_api_key=None)
        assert settings.describe()["embed_api_key"] is None

    def test_describe_does_not_contain_raw_key_anywhere(self) -> None:
        secret = "sk-super-secret-value-123456"
        settings = Settings(embed_api_key=secret)
        assert secret not in str(settings.describe())
