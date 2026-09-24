"""构建检索索引：把 PDF 变成可检索的向量索引。

用法示例::

    # 用离线 embedder 快速验证链路（不需要任何 API Key）
    uv run python scripts/ingest.py --input samples --output data/index

    # 用真实向量化模型（需先配置 LIT_RAG_EMBED_API_KEY）
    uv run python scripts/ingest.py --input data/raw --output data/index \
        --provider openai --model BAAI/bge-m3 --base-url https://api.siliconflow.cn/v1
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace

from dotenv import load_dotenv

from lit_rag.config import Settings
from lit_rag.embedding import create_embedder
from lit_rag.pipeline import ingest

logger = logging.getLogger("ingest")

DEFAULT_INPUT = "data/raw"
DEFAULT_OUTPUT = "data/index"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="把 PDF 文件/目录构建成检索索引",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        action="append",
        default=None,
        help=f"PDF 文件或目录，可重复指定（默认 {DEFAULT_INPUT}）",
    )
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT, help="索引输出目录")
    parser.add_argument(
        "--provider", choices=["offline", "openai"], default=None, help="向量化后端"
    )
    parser.add_argument("--model", default=None, help="向量化模型名，如 BAAI/bge-m3")
    parser.add_argument("--base-url", default=None, help="向量化服务地址")
    parser.add_argument("--chunk-size", type=int, default=None, help="chunk 长度上限（字符）")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="相邻 chunk 重叠字符数")
    parser.add_argument(
        "--drop-references",
        action="store_true",
        default=None,
        help="切分前裁掉参考文献列表",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")
    return parser


def settings_from_args(args: argparse.Namespace) -> Settings:
    """以环境变量为基底，用命令行参数覆盖。"""
    overrides: dict[str, object] = {}
    if args.provider:
        overrides["embed_provider"] = args.provider
    if args.model:
        overrides["embed_model"] = args.model
    if args.base_url:
        overrides["embed_base_url"] = args.base_url
    if args.chunk_size is not None:
        overrides["chunk_size"] = args.chunk_size
    if args.chunk_overlap is not None:
        overrides["chunk_overlap"] = args.chunk_overlap
    if args.drop_references is not None:
        overrides["drop_references"] = args.drop_references
    return replace(Settings.from_env(), **overrides)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(message)s",
    )

    inputs = args.input or [DEFAULT_INPUT]

    try:
        settings = settings_from_args(args)
    except ValueError as exc:
        logger.error("配置有误：%s", exc)
        return 2

    logger.info("配置：%s", settings.describe())

    try:
        embedder = create_embedder(settings)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2

    try:
        store = ingest(inputs, settings=settings, embedder=embedder)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1

    store.save(
        args.output,
        extra_meta={
            "embedder": embedder.name,
            "embed_provider": settings.embed_provider,
            "embed_model": settings.embed_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        },
    )

    print()
    print("索引构建完成")
    print(f"  chunk 数量 : {store.size}")
    print(f"  向量维度   : {store.dim}")
    print(f"  embedder   : {embedder.name}")
    print(f"  输出目录   : {args.output}")
    print()
    print("下一步：uv run python scripts/search.py '你的问题'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
