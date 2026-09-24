"""向量化（embedding）抽象层。

所有 embedder 都遵守同一个约定：

- 输入一组文本，输出形状为 ``(len(texts), dim)`` 的 ``float32`` 矩阵
- 输出矩阵**按行做 L2 归一化**，这样后续用点积即可得到余弦相似度
- 相同输入必须得到相同输出（调用方据此做缓存与测试）
"""

from __future__ import annotations

import abc
from collections.abc import Sequence

import numpy as np


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """按行做 L2 归一化；全零行保持为零向量（不会产生 NaN）。"""
    if matrix.size == 0:
        return matrix.astype(np.float32, copy=False)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (matrix / norms).astype(np.float32, copy=False)


def as_vector(matrix: np.ndarray, row: int = 0) -> np.ndarray:
    """取矩阵的第 ``row`` 行并保证是 1-D 向量。"""
    return np.asarray(matrix[row], dtype=np.float32).reshape(-1)


class Embedder(abc.ABC):
    """向量化器接口。"""

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        """向量维度。"""

    @property
    def name(self) -> str:
        """用于日志与索引元数据的可读名称。"""
        return type(self).__name__

    @abc.abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """把一批文本转成向量矩阵，形状 ``(len(texts), dim)``，行已 L2 归一化。"""

    def embed_one(self, text: str) -> np.ndarray:
        """把单条文本转成向量。"""
        matrix = self.embed([text])
        if matrix.shape[0] != 1:
            raise RuntimeError(f"{self.name}.embed() 返回的行数异常：{matrix.shape}")
        return as_vector(matrix, 0)

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"{type(self).__name__}(dim={self.dim})"
