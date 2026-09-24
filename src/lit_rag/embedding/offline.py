"""离线确定性向量化实现（hashing trick）。

**用途**：让整条 RAG 流水线在没有网络、没有 API Key 的环境下也能跑通并被测试。
GitHub Actions 的公开 CI 就靠它来验证链路正确性。

**原理**：把 ASCII 词元与字符 n-gram 用 ``blake2b`` 哈希到固定维度，
哈希最高位决定符号（signed hashing trick），词频做 sublinear 缩放
（``1 + log(tf)``），最后 L2 归一化。

**为什么用 blake2b 而不是内置 ``hash()``**：Python 内置 ``hash()`` 对字符串
会加随机盐（``PYTHONHASHSEED``），跨进程结果不一致，无法复现。哈希必须走
稳定的摘要算法。

.. warning::
   检索质量远低于真实 embedding 模型，**不要用于生产检索**。
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Iterator, Sequence

import numpy as np

from lit_rag.embedding.base import Embedder, l2_normalize

_WORD = re.compile(r"[a-z0-9]+")
_DIGEST_BYTES = 8
_SIGN_BIT = 63


class HashingEmbedder(Embedder):
    """基于哈希技巧的离线向量化器。"""

    def __init__(self, dim: int = 1024, ngram_range: tuple[int, int] = (2, 4)) -> None:
        """
        Args:
            dim: 向量维度。
            ngram_range: 字符 n-gram 的范围 ``(最小, 最大)``，闭区间。
        """
        if dim <= 0:
            raise ValueError(f"dim 必须为正整数，当前为 {dim}")
        low, high = ngram_range
        if low <= 0 or high < low:
            raise ValueError(f"ngram_range 非法：{ngram_range}")
        self._dim = dim
        self._ngram_range = (low, high)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        low, high = self._ngram_range
        return f"offline-hashing(dim={self._dim}, ngram={low}-{high})"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts = Counter(self._features(text))
            for feature, term_freq in counts.items():
                index, sign = self._bucket(feature)
                vectors[row, index] += sign * (1.0 + math.log(term_freq))
        return l2_normalize(vectors)

    # ------------------------------------------------------------------ 内部
    def _features(self, text: str) -> Iterator[str]:
        """产出特征串：ASCII 词元 + 字符 n-gram（后者对中文尤其重要）。"""
        lowered = text.lower()
        for match in _WORD.finditer(lowered):
            yield f"w:{match.group(0)}"
        low, high = self._ngram_range
        for size in range(low, high + 1):
            for start in range(len(lowered) - size + 1):
                gram = lowered[start : start + size]
                if gram.strip():
                    yield f"g{size}:{gram}"

    def _bucket(self, feature: str) -> tuple[int, float]:
        """把特征映射到 ``(下标, 符号)``。"""
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=_DIGEST_BYTES).digest()
        value = int.from_bytes(digest, "big")
        index = value % self._dim
        sign = 1.0 if (value >> _SIGN_BIT) & 1 else -1.0
        return index, sign
