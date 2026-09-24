"""通过 OpenAI 兼容接口调用真实向量化模型。

适用于任何实现了 ``POST /v1/embeddings`` 的服务，例如：

- **硅基流动 SiliconFlow** —— ``BAAI/bge-m3``（有免费额度，中文表现好）
- **智谱 AI** —— ``embedding-3``
- **阿里云百炼** —— ``text-embedding-v3`` / ``v4``
- **OpenAI** —— ``text-embedding-3-small``
- **本地部署** —— vLLM / Ollama / Xinference 的 OpenAI 兼容端点

.. note::
   DeepSeek 只提供对话模型，**不提供 embeddings 接口**，
   因此向量化必须单独配置一家服务商（对话仍可继续用 DeepSeek）。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from lit_rag.embedding.base import Embedder, l2_normalize

logger = logging.getLogger(__name__)


class OpenAICompatEmbedder(Embedder):
    """OpenAI 兼容的向量化器，自动分批、自动排序、自动归一化。"""

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        base_url: str | None = None,
        batch_size: int = 32,
        timeout: float = 60.0,
        max_retries: int = 3,
        dimensions: int | None = None,
    ) -> None:
        """
        Args:
            model: 模型名，例如 ``BAAI/bge-m3``。
            api_key: 服务商的 API Key。
            base_url: 接口地址，例如 ``https://api.siliconflow.cn/v1``。
            batch_size: 单次请求携带的文本条数。
            timeout: 单次请求超时（秒）。
            max_retries: SDK 层重试次数。
            dimensions: 部分模型支持的自定义输出维度。
        """
        if not model:
            raise ValueError("必须指定 embed_model")
        if not api_key:
            raise ValueError(
                "缺少向量化服务的 API Key。请设置 LIT_RAG_EMBED_API_KEY，"
                "或在 .env 中配置后重试。注意 DeepSeek 不提供 embeddings 接口。"
            )
        if batch_size <= 0:
            raise ValueError(f"batch_size 必须为正整数，当前为 {batch_size}")

        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._batch_size = batch_size
        self._timeout = timeout
        self._max_retries = max_retries
        self._dimensions = dimensions
        self._client = None
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            raise RuntimeError(
                "向量维度尚未确定：请先调用一次 embed()，或显式传入 dimensions。"
            )
        if self._dimensions:
            return self._dimensions
        return self._dim

    @property
    def name(self) -> str:
        return f"openai-compat({self._model})"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        items = list(texts)
        if not items:
            return np.zeros((0, self.dim), dtype=np.float32)

        batches: list[np.ndarray] = []
        for start in range(0, len(items), self._batch_size):
            batch = items[start : start + self._batch_size]
            logger.debug(
                "向量化批次 %d-%d / %d", start + 1, start + len(batch), len(items)
            )
            batches.append(self._embed_batch(batch))

        return l2_normalize(np.vstack(batches))

    def _embed_batch(self, batch: list[str]) -> np.ndarray:
        client = self._get_client()
        kwargs: dict[str, object] = {"model": self._model, "input": batch}
        if self._dimensions:
            kwargs["dimensions"] = self._dimensions

        response = client.embeddings.create(**kwargs)
        # 服务端不保证顺序，按 index 排回来
        ordered = sorted(response.data, key=lambda item: item.index)
        matrix = np.asarray([item.embedding for item in ordered], dtype=np.float32)

        if matrix.shape[0] != len(batch):
            raise RuntimeError(
                f"服务端返回条数不匹配：期望 {len(batch)}，实际 {matrix.shape[0]}"
            )
        self._dim = int(matrix.shape[1])
        return matrix

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout,
                max_retries=self._max_retries,
            )
        return self._client
