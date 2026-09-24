"""轻量向量库：余弦相似度检索 + 磁盘持久化。

之所以先用内存实现而不是直接上 Chroma / Milvus：

1. **零额外依赖** —— 只用 numpy，CI 与本地都不会因为向量库装不上而卡住
2. **链路可见** —— 检索逻辑完全透明，便于理解 RAG 的最小闭环
3. **接口一致** —— ``add`` / ``search`` / ``save`` / ``load`` 与主流向量库同形，
   将来替换成 Chroma 或 Milvus 时上层代码几乎不用改

索引规模到十万级 chunk 之前，这个实现都够用。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from lit_rag.schema import Chunk, Hit

logger = logging.getLogger(__name__)

VECTORS_FILE = "vectors.npy"
CHUNKS_FILE = "chunks.jsonl"
META_FILE = "meta.json"
FORMAT_VERSION = 1


def chunk_to_dict(chunk: Chunk) -> dict[str, object]:
    """把 Chunk 转成可 JSON 序列化的字典。"""
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "text": chunk.text,
        "index": chunk.index,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "section": chunk.section,
        "metadata": chunk.metadata,
    }


def chunk_from_dict(data: Mapping[str, object]) -> Chunk:
    """从字典还原 Chunk。"""
    raw_metadata = data.get("metadata")
    return Chunk(
        chunk_id=str(data["chunk_id"]),
        doc_id=str(data["doc_id"]),
        text=str(data["text"]),
        index=int(data.get("index", 0)),  # type: ignore[arg-type]
        page_start=int(data.get("page_start", 0)),  # type: ignore[arg-type]
        page_end=int(data.get("page_end", 0)),  # type: ignore[arg-type]
        section=str(data.get("section", "")),
        metadata=dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {},
    )


class VectorStore:
    """内存向量索引。

    向量在入库前应已 L2 归一化，因此检索退化为矩阵乘法（点积 = 余弦相似度）。
    """

    def __init__(self, dim: int | None = None) -> None:
        self._dim = dim
        self._vectors: np.ndarray | None = None
        self._chunks: list[Chunk] = []
        self._meta: dict[str, object] = {}

    # ------------------------------------------------------------- 只读属性
    @property
    def size(self) -> int:
        """已入库的 chunk 数量。"""
        return len(self._chunks)

    @property
    def dim(self) -> int:
        """向量维度。索引为空且未指定时为 0。"""
        if self._vectors is not None:
            return int(self._vectors.shape[1])
        return self._dim or 0

    @property
    def is_empty(self) -> bool:
        return not self._chunks

    @property
    def meta(self) -> dict[str, object]:
        """索引元数据（来自 ``meta.json``，供调用方校验 embedder 是否匹配）。"""
        return dict(self._meta)

    def chunks(self) -> list[Chunk]:
        """返回 chunk 列表的浅拷贝。"""
        return list(self._chunks)

    # ----------------------------------------------------------------- 写入
    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        """批量写入 chunk 及其向量。

        Args:
            chunks: chunk 列表。
            vectors: 形状 ``(len(chunks), dim)`` 的向量矩阵。

        Raises:
            ValueError: 行数或维度不匹配。
        """
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2:
            raise ValueError(f"vectors 必须是二维矩阵，当前形状为 {matrix.shape}")
        if matrix.shape[0] != len(chunks):
            raise ValueError(
                f"chunk 数量({len(chunks)}) 与 向量行数({matrix.shape[0]}) 不一致"
            )
        if not chunks:
            return

        self._dim = int(matrix.shape[1])
        if self._vectors is None:
            self._vectors = matrix
            self._chunks = list(chunks)
            return

        if int(self._vectors.shape[1]) != self._dim:
            raise ValueError(
                f"向量维度不一致：索引为 {self._vectors.shape[1]}，新数据为 {self._dim}"
            )
        self._vectors = np.vstack([self._vectors, matrix])
        self._chunks.extend(chunks)

    # ----------------------------------------------------------------- 检索
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        *,
        min_score: float | None = None,
    ) -> list[Hit]:
        """按余弦相似度检索最相近的 chunk。

        Args:
            query_vector: 查询向量（应已 L2 归一化）。
            top_k: 返回条数。
            min_score: 低于该分数的结果会被过滤。

        Returns:
            按分数降序排列的 :class:`~lit_rag.schema.Hit` 列表。
        """
        if top_k <= 0:
            raise ValueError(f"top_k 必须为正整数，当前为 {top_k}")
        if self.is_empty or self._vectors is None:
            return []

        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != self._vectors.shape[1]:
            raise ValueError(
                f"查询向量维度({query.shape[0]}) 与索引维度({self._vectors.shape[1]}) 不一致"
            )

        scores = self._vectors @ query
        # argpartition 先取前 k 个候选（O(n)），再对这 k 个精确排序
        k = min(top_k, scores.shape[0])
        candidates = np.argpartition(-scores, k - 1)[:k]
        ranked = sorted(candidates.tolist(), key=lambda i: float(scores[i]), reverse=True)

        hits: list[Hit] = []
        for idx in ranked:
            score = float(scores[idx])
            if min_score is not None and score < min_score:
                continue
            hits.append(Hit(chunk=self._chunks[idx], score=score))
        return hits

    # ------------------------------------------------------------- 持久化
    def save(
        self,
        directory: str | Path,
        *,
        extra_meta: Mapping[str, object] | None = None,
    ) -> Path:
        """把索引写入目录。

        Args:
            directory: 目标目录，不存在会自动创建。
            extra_meta: 附加元数据，建议写入 embedder 名称，便于检索时校验一致性。
        """
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)

        vectors = self._vectors
        if vectors is None:
            vectors = np.zeros((0, self.dim), dtype=np.float32)

        np.save(str(target / VECTORS_FILE), vectors)
        with (target / CHUNKS_FILE).open("w", encoding="utf-8", newline="\n") as handle:
            for chunk in self._chunks:
                handle.write(json.dumps(chunk_to_dict(chunk), ensure_ascii=False) + "\n")

        meta: dict[str, object] = {
            "format_version": FORMAT_VERSION,
            "dim": int(vectors.shape[1]),
            "size": len(self._chunks),
            "created_at": datetime.now(UTC).isoformat(),
        }
        if extra_meta:
            meta.update(extra_meta)
        self._meta = meta

        (target / META_FILE).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("索引已保存到 %s（%d 条，维度 %d）", target, meta["size"], meta["dim"])
        return target

    @classmethod
    def load(cls, directory: str | Path) -> VectorStore:
        """从目录读取索引。

        Raises:
            FileNotFoundError: 目录或必需文件不存在。
            ValueError: 索引文件内容不一致（可能被手工改坏）。
        """
        source = Path(directory)
        vectors_path = source / VECTORS_FILE
        chunks_path = source / CHUNKS_FILE
        if not vectors_path.is_file() or not chunks_path.is_file():
            raise FileNotFoundError(
                f"索引不完整，缺少 {VECTORS_FILE} 或 {CHUNKS_FILE}：{source}"
            )

        vectors = np.load(str(vectors_path)).astype(np.float32)
        chunks = [
            chunk_from_dict(json.loads(line))
            for line in chunks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        if vectors.shape[0] != len(chunks):
            raise ValueError(
                f"索引损坏：向量行数({vectors.shape[0]}) 与 chunk 数({len(chunks)}) 不一致"
            )

        store = cls(dim=int(vectors.shape[1]) if vectors.ndim == 2 else None)
        store._vectors = vectors if vectors.shape[0] else None
        store._chunks = chunks

        meta_path = source / META_FILE
        if meta_path.is_file():
            loaded = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                store._meta = loaded
                version = loaded.get("format_version")
                if version != FORMAT_VERSION:
                    logger.warning(
                        "索引格式版本为 %s，当前代码期望 %s，建议重新构建索引",
                        version,
                        FORMAT_VERSION,
                    )
        return store

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"VectorStore(size={self.size}, dim={self.dim})"
