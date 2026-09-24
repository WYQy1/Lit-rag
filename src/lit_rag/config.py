"""运行配置：统一从环境变量读取，方便本地开发与 CI 复用同一套代码。

所有环境变量都以 ``LIT_RAG_`` 为前缀，避免与系统变量冲突。
CLI 脚本通过 :meth:`Settings.describe` 打印配置时会自动脱敏 API Key。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

ENV_PREFIX = "LIT_RAG_"

#: 可选的向量化后端
EMBED_PROVIDERS = ("offline", "openai")
#: 可选的 PDF 解析后端
PDF_BACKENDS = ("auto", "pymupdf", "pypdf")


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(f"{ENV_PREFIX}{name}")
    return raw.strip() if raw and raw.strip() else default


def _env_opt_str(name: str) -> str | None:
    raw = os.getenv(f"{ENV_PREFIX}{name}")
    return raw.strip() if raw and raw.strip() else None


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(f"{ENV_PREFIX}{name}")
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"环境变量 {ENV_PREFIX}{name} 必须是整数，当前值为 {raw!r}") from exc


@dataclass(slots=True)
class Settings:
    """流水线配置。"""

    # --- 切分 ---
    chunk_size: int = 800
    chunk_overlap: int = 120
    min_chunk_size: int = 80

    # --- PDF 解析 ---
    pdf_backend: str = "auto"
    drop_references: bool = False

    # --- 向量化 ---
    embed_provider: str = "offline"
    embed_model: str = "BAAI/bge-m3"
    embed_base_url: str | None = None
    embed_api_key: str | None = None
    embed_batch_size: int = 32

    # --- 检索 ---
    top_k: int = 5

    extra: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size 必须为正整数")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap 不能为负数")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap({self.chunk_overlap}) 必须小于 chunk_size({self.chunk_size})"
            )
        if self.embed_provider not in EMBED_PROVIDERS:
            raise ValueError(
                f"embed_provider 必须是 {EMBED_PROVIDERS} 之一，当前为 {self.embed_provider!r}"
            )
        if self.pdf_backend not in PDF_BACKENDS:
            raise ValueError(
                f"pdf_backend 必须是 {PDF_BACKENDS} 之一，当前为 {self.pdf_backend!r}"
            )

    @classmethod
    def from_env(cls) -> Settings:
        """从环境变量构造配置（缺失项使用默认值）。"""
        return cls(
            chunk_size=_env_int("CHUNK_SIZE", 800),
            chunk_overlap=_env_int("CHUNK_OVERLAP", 120),
            min_chunk_size=_env_int("MIN_CHUNK_SIZE", 80),
            pdf_backend=_env_str("PDF_BACKEND", "auto"),
            drop_references=_env_str("DROP_REFERENCES", "false").lower() in {"1", "true", "yes"},
            embed_provider=_env_str("EMBED_PROVIDER", "offline"),
            embed_model=_env_str("EMBED_MODEL", "BAAI/bge-m3"),
            embed_base_url=_env_opt_str("EMBED_BASE_URL"),
            embed_api_key=_env_opt_str("EMBED_API_KEY"),
            embed_batch_size=_env_int("EMBED_BATCH_SIZE", 32),
            top_k=_env_int("TOP_K", 5),
        )

    def describe(self) -> dict[str, object]:
        """返回可安全打印的配置快照（API Key 已脱敏）。"""
        masked: str | None = None
        if key := self.embed_api_key:
            masked = f"{key[:6]}...{key[-4:]}" if len(key) > 12 else "***"
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "min_chunk_size": self.min_chunk_size,
            "pdf_backend": self.pdf_backend,
            "drop_references": self.drop_references,
            "embed_provider": self.embed_provider,
            "embed_model": self.embed_model,
            "embed_base_url": self.embed_base_url,
            "embed_api_key": masked,
            "embed_batch_size": self.embed_batch_size,
            "top_k": self.top_k,
        }
