"""向量化模块：统一入口与工厂函数。"""

from __future__ import annotations

import os

from lit_rag.config import Settings
from lit_rag.embedding.base import Embedder, l2_normalize
from lit_rag.embedding.offline import HashingEmbedder
from lit_rag.embedding.openai_compat import OpenAICompatEmbedder

__all__ = [
    "Embedder",
    "HashingEmbedder",
    "OpenAICompatEmbedder",
    "create_embedder",
    "l2_normalize",
]

#: 在未显式配置 LIT_RAG_EMBED_API_KEY 时，依次回退尝试这些环境变量
_API_KEY_FALLBACKS = (
    "EMBED_API_KEY",
    "SILICONFLOW_API_KEY",
    "ZHIPUAI_API_KEY",
    "DASHSCOPE_API_KEY",
    "OPENAI_API_KEY",
)

#: 各家服务商常见的 base_url（按需在 .env 中显式覆盖）
DEFAULT_BASE_URLS = {
    "siliconflow": "https://api.siliconflow.cn/v1",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "openai": "https://api.openai.com/v1",
}


def _resolve_api_key(settings: Settings) -> str | None:
    if settings.embed_api_key:
        return settings.embed_api_key
    for name in _API_KEY_FALLBACKS:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return None


def create_embedder(settings: Settings) -> Embedder:
    """按配置构造 embedder。

    Args:
        settings: 运行配置。

    Returns:
        ``offline`` 返回 :class:`HashingEmbedder`，
        ``openai`` 返回 :class:`OpenAICompatEmbedder`。

    Raises:
        ValueError: provider 未知，或缺少必要的 API Key。
    """
    if settings.embed_provider == "offline":
        return HashingEmbedder()

    if settings.embed_provider == "openai":
        api_key = _resolve_api_key(settings)
        if not api_key:
            tried = ", ".join(_API_KEY_FALLBACKS)
            raise ValueError(
                "向量化服务未配置 API Key。请设置 LIT_RAG_EMBED_API_KEY，"
                f"或提供以下任一环境变量：{tried}。\n"
                "提示：DeepSeek 只提供对话模型，不提供 embeddings 接口，"
                "需要单独配置一家向量化服务商（如硅基流动的 BAAI/bge-m3）。"
            )
        return OpenAICompatEmbedder(
            model=settings.embed_model,
            api_key=api_key,
            base_url=settings.embed_base_url,
            batch_size=settings.embed_batch_size,
        )

    raise ValueError(f"未知的 embed_provider：{settings.embed_provider!r}")
