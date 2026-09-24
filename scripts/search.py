"""检索索引：给定问题，找出最相关的文献片段。

用法示例::

    uv run python scripts/search.py "Cr3+ 掺杂浓度对发射峰位有什么影响" --top-k 5

输出会带上出处（文件名 / 页码 / 章节），方便人工核对。
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace

from dotenv import load_dotenv

from lit_rag.config import Settings
from lit_rag.embedding import create_embedder
from lit_rag.pipeline import format_hits, retrieve
from lit_rag.store import VectorStore

logger = logging.getLogger("search")

DEFAULT_INDEX = "data/index"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="在已构建的索引中检索文献片段",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("query", help="查询问题")
    parser.add_argument("-i", "--index", default=DEFAULT_INDEX, help="索引目录")
    parser.add_argument("-k", "--top-k", type=int, default=None, help="返回条数")
    parser.add_argument(
        "--provider", choices=["offline", "openai"], default=None, help="向量化后端"
    )
    parser.add_argument("--model", default=None, help="向量化模型名")
    parser.add_argument("--base-url", default=None, help="向量化服务地址")
    parser.add_argument(
        "--min-score", type=float, default=None, help="过滤低于该相似度的结果"
    )
    parser.add_argument("--full", action="store_true", help="打印完整 chunk 文本")
    parser.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")
    return parser


def settings_from_args(args: argparse.Namespace) -> Settings:
    overrides: dict[str, object] = {}
    if args.provider:
        overrides["embed_provider"] = args.provider
    if args.model:
        overrides["embed_model"] = args.model
    if args.base_url:
        overrides["embed_base_url"] = args.base_url
    if args.top_k is not None:
        overrides["top_k"] = args.top_k
    return replace(Settings.from_env(), **overrides)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)-7s %(message)s",
    )

    try:
        settings = settings_from_args(args)
    except ValueError as exc:
        logger.error("配置有误：%s", exc)
        return 2

    try:
        store = VectorStore.load(args.index)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        logger.error("提示：先运行 uv run python scripts/ingest.py 构建索引")
        return 1

    try:
        embedder = create_embedder(settings)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2

    # 索引与查询必须使用同一个 embedder，否则分数毫无意义 —— 这里做一次校验
    indexed_embedder = store.meta.get("embedder")
    if indexed_embedder and indexed_embedder != embedder.name:
        logger.warning(
            "embedder 不一致！索引由 %r 构建，当前使用的是 %r，检索结果不可信。",
            indexed_embedder,
            embedder.name,
        )

    try:
        hits = retrieve(
            args.query,
            store,
            embedder,
            top_k=settings.top_k,
            min_score=args.min_score,
        )
    except ValueError as exc:
        logger.error("%s", exc)
        return 2

    print(f"查询：{args.query}")
    print(f"索引：{args.index}（{store.size} 个 chunk，{store.dim} 维）")
    print(f"embedder：{embedder.name}")
    print("-" * 60)
    if args.full:
        for rank, hit in enumerate(hits, start=1):
            print(f"\n[{rank}] score={hit.score:.4f}  {hit.chunk.citation()}\n")
            print(hit.chunk.text)
    else:
        print(format_hits(hits))
    print("-" * 60)
    print(f"共 {len(hits)} 条结果")
    return 0


if __name__ == "__main__":
    sys.exit(main())
