"""结构感知的文本切分。

与「按固定长度暴力截断」相比，这里做了三件事：

1. **段落优先** —— 先在自然段边界聚合，避免把一句话劈成两半
2. **章节感知** —— 识别 ``1. Introduction`` / ``2.1 Sample preparation`` 这类标题，
   写进 chunk 元数据；换章节时强制断开，保证一个 chunk 不跨章
3. **超长段落兜底** —— 段落本身超过阈值时，退化为按句子切分，再退化为硬切

``chunk_size`` 的单位是**字符数**。经验换算：英文约 4 字符/token，
中文约 1 字符/token，因此 800 字符大致相当于 200 个英文单词。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

from lit_rag.schema import Chunk, Document

# 编号标题：1. Introduction / 2.1 Sample preparation / 3.2.1 Results
# 要求标题正文以大写字母或中文字符开头，避免把 "3.5 g of powder." 误判为标题
_HEADING = re.compile(r"^\s*(\d+(?:\.\d+){0,3})[.)]?\s+([A-Z\u4e00-\u9fff][^\n]{0,80})$")

# 无编号标题
_PLAIN_HEADING = re.compile(
    r"^\s*(abstract|introduction|experimental|experimental section|methods?|materials? and methods?|"
    r"results?(?: and discussion)?|discussion|conclusions?|summary|"
    r"acknowledg(?:e)?ments?|references|bibliography|supporting information)\s*$",
    re.IGNORECASE,
)

# 句子边界：中文标点后，或英文句点 + 空白
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？；!?;])\s*|(?<=\.)\s+")

# 中日韩字符（含中文标点与全角符号），用于判断拼接时要不要补空格
_CJK = re.compile(r"[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]")

_MAX_HEADING_LEN = 90


@dataclass(slots=True)
class _Block:
    """切分的最小输入单元：一个自然段（或超长段落切出来的一段）。"""

    text: str
    page: int
    section: str


@dataclass(slots=True)
class _Piece:
    """聚合后的候选 chunk。"""

    text: str
    page_start: int
    page_end: int
    section: str


def _is_heading(line: str) -> bool:
    """判断一行是不是章节标题。"""
    stripped = line.strip()
    if not stripped or len(stripped) > _MAX_HEADING_LEN:
        return False
    if _PLAIN_HEADING.match(stripped):
        return True
    # 标题极少以句号结尾，借此排除普通句子
    if stripped.endswith("."):
        return False
    return bool(_HEADING.match(stripped))


def _normalize_heading(line: str) -> str:
    return line.strip().rstrip(".").strip()


def _split_long_text(text: str, chunk_size: int) -> list[str]:
    """把超长文本按句子边界切开；单句仍超长时硬切。"""
    sentences = [s for s in _SENTENCE_SPLIT.split(text) if s and s.strip()]
    if not sentences:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    pieces: list[str] = []
    buffer = ""
    for sentence in sentences:
        if len(sentence) > chunk_size:
            if buffer:
                pieces.append(buffer)
                buffer = ""
            pieces.extend(
                sentence[i : i + chunk_size] for i in range(0, len(sentence), chunk_size)
            )
            continue
        candidate = f"{buffer} {sentence}".strip() if buffer else sentence
        if len(candidate) <= chunk_size:
            buffer = candidate
        else:
            pieces.append(buffer)
            buffer = sentence
    if buffer:
        pieces.append(buffer)
    return [p for p in pieces if p.strip()]


def _joined(lines: list[str]) -> str:
    """把同一段落的多个物理行拼回一行。

    中文之间直接拼接，英文之间补一个空格 —— 否则会得到
    「这是 一 句话」这种被空格打散的中文。

    之所以需要这一步：PDF 抽取出来的文本几乎没有空行，
    一段话会被拆成很多个物理行；必须先把它们接回去，
    否则句子切分会失效、章节标题也会被埋在段落中间。
    """
    result = ""
    for line in lines:
        if not result:
            result = line
            continue
        # 只有「两侧都是中文」时才直接拼接；中英混排要补空格，
        # 否则会得到 "emission maximum.摘要：..." 这种粘连。
        if _CJK.search(result[-1]) and _CJK.search(line[0]):
            result += line
        else:
            result += " " + line
    return result


def _iter_blocks(doc: Document) -> list[_Block]:
    """把文档拆成「段落 + 所属章节 + 页码」的序列。

    逐行扫描而不是按空行切分：PDF 抽取结果里空行极不可靠，
    标题往往紧跟在正文行之间。遇到标题就切换当前章节，
    遇到空行就强制断开段落。
    """
    blocks: list[_Block] = []
    current_section = ""
    buffer: list[str] = []
    buffer_page = 0

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        text = _joined(buffer).strip()
        if text:
            blocks.append(_Block(text=text, page=buffer_page, section=current_section))
        buffer = []

    for page in doc.pages:
        for raw_line in page.text.split("\n"):
            line = raw_line.strip()
            if not line:
                flush()
                continue
            if _is_heading(line):
                flush()
                current_section = _normalize_heading(line)
                continue
            if not buffer:
                buffer_page = page.number
            buffer.append(line)

    flush()
    return blocks


def _expand_oversized(blocks: list[_Block], chunk_size: int) -> list[_Block]:
    """把超过 chunk_size 的 block 拆成多个不超过它的 block。"""
    expanded: list[_Block] = []
    for block in blocks:
        if len(block.text) <= chunk_size:
            expanded.append(block)
            continue
        for piece in _split_long_text(block.text, chunk_size):
            expanded.append(_Block(text=piece, page=block.page, section=block.section))
    return expanded


def _pack(blocks: list[_Block], chunk_size: int, overlap: int) -> list[_Piece]:
    """把 block 贪心聚合成 chunk，并在相邻 chunk 之间保留重叠。"""
    pieces: list[_Piece] = []
    buffer = ""
    page_start = 0
    page_end = 0
    section = ""

    def flush() -> None:
        nonlocal buffer
        if buffer.strip():
            pieces.append(
                _Piece(
                    text=buffer.strip(),
                    page_start=page_start,
                    page_end=page_end,
                    section=section,
                )
            )
        buffer = ""

    for block in blocks:
        if not buffer:
            buffer, page_start, page_end, section = block.text, block.page, block.page, block.section
            continue

        candidate_len = len(buffer) + 2 + len(block.text)
        if block.section == section and candidate_len <= chunk_size:
            buffer = f"{buffer}\n\n{block.text}"
            page_end = block.page
            continue

        # 需要断开：先收尾，再用上一块的尾部作为重叠种子开新块。
        # 注意：换章节时不加重叠 —— 否则上一章的内容会被标成新章节，污染引用信息。
        flush()
        seed = ""
        if overlap > 0 and pieces and block.section == section:
            available = chunk_size - len(block.text) - 2
            if available > 0:
                seed = _tail_seed(prev_piece=pieces[-1].text, overlap=overlap, limit=available)
        buffer = f"{seed}\n\n{block.text}" if seed else block.text
        page_start, page_end, section = block.page, block.page, block.section

    flush()
    return pieces


def _tail_seed(*, prev_piece: str, overlap: int, limit: int) -> str:
    """从上一块尾部截取重叠文本，且不超过 ``limit`` 个字符。"""
    take = min(overlap, limit, len(prev_piece))
    if take <= 0:
        return ""
    tail = prev_piece[-take:]
    # 尽量从词边界开始，避免把单词劈开
    space = tail.find(" ")
    if 0 < space < 20:
        tail = tail[space + 1 :]
    return tail.strip()


def _merge_tiny(pieces: list[_Piece], chunk_size: int, min_chunk_size: int) -> list[_Piece]:
    """把过短的碎片并回前一块，避免产生只有一两句话的噪声 chunk。

    **不跨章节合并** —— 否则「2. Methods」这种内容很少的小节会被并进
    上一节，章节标签随之丢失，引用信息就不准了。
    """
    merged: list[_Piece] = []
    for piece in pieces:
        if (
            merged
            and len(piece.text) < min_chunk_size
            and piece.section == merged[-1].section
            and len(merged[-1].text) + 2 + len(piece.text) <= chunk_size
        ):
            previous = merged[-1]
            merged[-1] = replace(
                previous,
                text=f"{previous.text}\n\n{piece.text}",
                page_end=piece.page_end,
            )
        else:
            merged.append(piece)
    return merged


def _to_chunks(doc: Document, pieces: list[_Piece]) -> list[Chunk]:
    filename = str(doc.metadata.get("filename") or Path(doc.source).name)
    chunks: list[Chunk] = []
    for index, piece in enumerate(pieces):
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}:{index:04d}",
                doc_id=doc.doc_id,
                text=piece.text,
                index=index,
                page_start=piece.page_start,
                page_end=piece.page_end,
                section=piece.section,
                metadata={
                    "source": filename,
                    "title": doc.title,
                    "n_chars": len(piece.text),
                },
            )
        )
    return chunks


def split_document(
    doc: Document,
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    min_chunk_size: int = 80,
) -> list[Chunk]:
    """把一份文档切分成 chunk 列表。

    Args:
        doc: 已解析的文档。
        chunk_size: 单个 chunk 的目标长度上限（字符数）。
        chunk_overlap: 相邻 chunk 的重叠长度（字符数）。
        min_chunk_size: 低于此长度的碎片会尽量并回前一块。

    Returns:
        按顺序排列的 :class:`~lit_rag.schema.Chunk` 列表。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正整数")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap 不能为负数")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")

    blocks = _iter_blocks(doc)
    if not blocks:
        return []

    units = _expand_oversized(blocks, chunk_size)
    pieces = _merge_tiny(_pack(units, chunk_size, chunk_overlap), chunk_size, min_chunk_size)
    return _to_chunks(doc, pieces)


def split_documents(
    docs: list[Document],
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    min_chunk_size: int = 80,
) -> list[Chunk]:
    """批量切分多份文档。"""
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(
            split_document(
                doc,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                min_chunk_size=min_chunk_size,
            )
        )
    return chunks
