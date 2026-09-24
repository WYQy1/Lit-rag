"""文本清洗：把 PDF 抽取出来的「脏文本」整理成适合切分的形态。

PDF 抽取的典型噪声：

1. 跨行断词 —— ``analy-\\nsis`` 应为 ``analysis``
2. 页眉 / 页脚 —— 期刊名、作者、页码在多页重复出现
3. 孤立的页码行 —— ``123`` 或 ``Page 4 of 12``
4. 多余空白 —— 排版造成的连续空格与空行
5. 参考文献列表 —— 会污染检索结果，可按需整体裁掉

清洗顺序很重要：**先去连字符，再做空白归一化**，否则换行信息会丢失。
"""

from __future__ import annotations

import re

# --- 跨行断词：analy-\nsis -> analysis（允许连字符后有空格） ---
_HYPHEN_BREAK = re.compile(r"(?<=[A-Za-z])-\s*\n\s*(?=[a-z])")

# --- 纯页码行 ---
_PAGE_NUMBER = re.compile(r"^\s*(?:page\s*)?\d{1,4}\s*(?:/\s*\d{1,4})?\s*$", re.IGNORECASE)
_ROMAN_PAGE = re.compile(r"^\s*[ivxlcdm]{1,7}\s*$", re.IGNORECASE)
_PAGE_OF = re.compile(r"^\s*page\s+\d+\s+of\s+\d+\s*$", re.IGNORECASE)

# --- 参考文献起始标题 ---
_REFERENCES_HEADING = re.compile(
    r"^\s*(?:\d+\.?\s*)?(references|bibliography|reference list|参考文献)\s*$",
    re.IGNORECASE,
)


def dehyphenate(text: str) -> str:
    """修复英文跨行断词。

    只处理「小写字母开头的续行」，避免误伤 ``Cr3+-\\nDoped`` 这类
    连字符本身就是内容一部分的情况。
    """
    return _HYPHEN_BREAK.sub("", text)


def normalize_whitespace(text: str) -> str:
    """折叠多余空格与空行，但保留段落边界（``\\n\\n``）。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u3000", " ")  # 不换行空格、全角空格
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_page_number(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return bool(
        _PAGE_NUMBER.match(stripped) or _ROMAN_PAGE.match(stripped) or _PAGE_OF.match(stripped)
    )


def strip_page_numbers(text: str) -> str:
    """删除只包含页码的行。"""
    lines = [line for line in text.split("\n") if not _is_page_number(line)]
    return "\n".join(lines)


def _edge_lines(text: str, n: int) -> list[str]:
    """取一页最前面和最后面的 n 个非空行。"""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return []
    return lines[:n] + lines[-n:]


def detect_repeated_lines(
    pages: list[str],
    *,
    edge_lines: int = 2,
    min_ratio: float = 0.6,
) -> set[str]:
    """识别跨页重复出现的页眉 / 页脚。

    统计每页首尾各 ``edge_lines`` 行的出现频次，频率达到
    ``min_ratio``（且至少出现 2 次）的行会被判定为页眉页脚。

    Args:
        pages: 每页的原始文本。
        edge_lines: 只看每页的首尾多少行。
        min_ratio: 判定阈值（占页数比例）。

    Returns:
        需要删除的行文本集合。
    """
    if len(pages) < 2:
        return set()

    counts: dict[str, int] = {}
    for page in pages:
        for line in set(_edge_lines(page, edge_lines)):
            counts[line] = counts.get(line, 0) + 1

    threshold = max(2, int(len(pages) * min_ratio + 0.999))
    return {line for line, count in counts.items() if count >= threshold}


def clean_page(text: str, *, repeated: set[str] | None = None) -> str:
    """清洗单页文本（不做跨页统计）。"""
    text = dehyphenate(text)
    if repeated:
        lines = [line for line in text.split("\n") if line.strip() not in repeated]
        text = "\n".join(lines)
    text = strip_page_numbers(text)
    return normalize_whitespace(text)


def clean_pages(
    pages: list[str],
    *,
    edge_lines: int = 2,
    min_ratio: float = 0.6,
    drop_references: bool = False,
) -> list[str]:
    """批量清洗多页文本。

    Args:
        pages: 每页的原始文本。
        edge_lines: 页眉页脚检测时每页考察的首尾行数。
        min_ratio: 页眉页脚判定阈值。
        drop_references: 是否裁掉参考文献列表（默认保留，因为参考文献
            有时也含有用信息；做纯正文 RAG 时可开启）。

    Returns:
        清洗后的每页文本，长度与输入一致。
    """
    repeated = detect_repeated_lines(pages, edge_lines=edge_lines, min_ratio=min_ratio)
    cleaned = [clean_page(page, repeated=repeated) for page in pages]
    if drop_references:
        cleaned = [split_references(page)[0] for page in cleaned]
    return cleaned


def split_references(text: str) -> tuple[str, str]:
    """把文本切成 ``(正文, 参考文献)`` 两段。

    若找不到参考文献标题，则返回 ``(原文本, "")``。
    """
    lines = text.split("\n")
    for idx, line in enumerate(lines):
        if _REFERENCES_HEADING.match(line.strip()):
            body = "\n".join(lines[:idx]).strip()
            refs = "\n".join(lines[idx:]).strip()
            return body, refs
    return text, ""
