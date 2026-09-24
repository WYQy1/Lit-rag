"""生成用于测试的样本 PDF。

为什么要「生成」而不是随便找一篇论文放进来：

1. **可复现** —— 任何人 clone 下来都能重新生成同一份夹具
2. **无版权风险** —— 内容是虚构的演示文本，可以安全提交到公开仓库
3. **针对性强** —— 刻意埋了页眉、页码、章节标题、参考文献等噪声，
   用来验证清洗与切分逻辑真的有效

生成结果：``samples/nir-phosphor-demo.pdf``

用法::

    uv run python scripts/make_sample_pdf.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pymupdf

PAGE_W, PAGE_H = 595.0, 842.0  # A4
MARGIN = 62.0
TOP = 72.0
BOTTOM = PAGE_H - 72.0

BODY = 10.0
H1 = 12.5
H2 = 11.0
TITLE = 15.0
LINE_GAP = 3.5

# 页眉在每页都出现 —— 用来验证「跨页重复行」检测
RUNNING_HEADER = "Journal of Luminescence Demo - Submitted Manuscript"

CJK_FONT = "china-s"


def text_width(text: str, fontname: str, fontsize: float) -> float:
    """估算字符串宽度；CJK 内置字体不支持测量时退化为估算。"""
    try:
        return float(pymupdf.get_text_length(text, fontname=fontname, fontsize=fontsize))
    except Exception:  # noqa: BLE001 - 字体不支持测量时走估算分支
        return len(text) * fontsize * 0.95


def break_token(token: str, fontname: str, fontsize: float, max_width: float) -> list[str]:
    """把超宽的单个 token 按字符切开（中文长句会走到这里）。"""
    parts: list[str] = []
    current = ""
    for char in token:
        trial = current + char
        if text_width(trial, fontname, fontsize) <= max_width:
            current = trial
        else:
            if current:
                parts.append(current)
            current = char
    if current:
        parts.append(current)
    return parts


def wrap(text: str, fontname: str, fontsize: float, max_width: float) -> list[str]:
    """按可用宽度折行。"""
    lines: list[str] = []
    for raw_line in text.split("\n"):
        words = raw_line.split(" ")
        current = ""
        for word in words:
            trial = f"{current} {word}".strip()
            if text_width(trial, fontname, fontsize) <= max_width:
                current = trial
                continue
            if current:
                lines.append(current)
                current = ""
            if text_width(word, fontname, fontsize) > max_width:
                pieces = break_token(word, fontname, fontsize, max_width)
                lines.extend(pieces[:-1])
                current = pieces[-1] if pieces else ""
            else:
                current = word
        if current:
            lines.append(current)
    return lines


class Builder:
    """极简的 PDF 排版器：顺序写入块级内容，自动翻页并加页眉页码。"""

    def __init__(self) -> None:
        self.doc = pymupdf.open()
        self.page: pymupdf.Page | None = None
        self.y = TOP
        self.page_no = 0
        self.new_page()

    def new_page(self) -> None:
        self.page = self.doc.new_page(width=PAGE_W, height=PAGE_H)
        self.page_no += 1
        assert self.page is not None
        self.page.insert_text(
            (MARGIN, 46.0),
            RUNNING_HEADER,
            fontname="helv",
            fontsize=7.5,
            color=(0.45, 0.45, 0.45),
        )
        # 页码单独一行，用来验证「纯页码行」清洗
        self.page.insert_text(
            (PAGE_W - MARGIN - 12.0, PAGE_H - 46.0),
            str(self.page_no),
            fontname="helv",
            fontsize=8.5,
            color=(0.45, 0.45, 0.45),
        )
        self.y = TOP

    def _ensure(self, height: float) -> None:
        if self.y + height > BOTTOM:
            self.new_page()

    def block(
        self,
        text: str,
        *,
        fontsize: float = BODY,
        fontname: str = "helv",
        gap_before: float = 0.0,
        gap_after: float = 6.0,
        max_width: float | None = None,
    ) -> None:
        width = max_width if max_width is not None else PAGE_W - 2 * MARGIN
        assert self.page is not None
        self.y += gap_before
        for line in wrap(text, fontname, fontsize, width):
            line_height = fontsize + LINE_GAP
            self._ensure(line_height)
            assert self.page is not None
            self.page.insert_text(
                (MARGIN, self.y + fontsize), line, fontname=fontname, fontsize=fontsize
            )
            self.y += line_height
        self.y += gap_after

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.doc.set_metadata(
            {
                "title": "Site-Selective Cr3+ Occupancy and Broadband Near-Infrared "
                "Emission in Garnet-Type Phosphors",
                "author": "Demo Author, Demo Coauthor",
                "subject": "Synthetic sample generated for Lit-rag tests",
            }
        )
        self.doc.save(str(path))
        self.doc.close()
        return path


def build() -> Path:
    b = Builder()

    b.block(
        "Site-Selective Cr3+ Occupancy and Broadband Near-Infrared "
        "Emission in Garnet-Type Phosphors",
        fontsize=TITLE,
        gap_after=8.0,
    )
    b.block(
        "Demo Author, Demo Coauthor, and Demo Corresponding Author*",
        fontsize=9.0,
        gap_after=4.0,
    )
    b.block(
        "*Key Laboratory of Extreme Luminescent Materials, Demo University, "
        "Zhengzhou 450002, China",
        fontsize=8.5,
        gap_after=12.0,
    )

    b.block("Abstract", fontsize=H1, gap_after=4.0)
    b.block(
        "Cr3+-activated garnet phosphors are promising broadband near-infrared (NIR) "
        "emitters for compact light sources. However, the relationship between "
        "site-selective Cr3+ occupancy and the resulting emission profile remains "
        "debated. Here we report a series of garnet-type phosphors in which the "
        "Cr3+ concentration is varied systematically, and we correlate the "
        "occupancy of octahedral versus tetrahedral sites with the observed "
        "redshift of the emission maximum.",
        gap_after=8.0,
    )

    # 中文摘要：用来验证清洗与切分对 CJK 文本的处理
    b.block(
        "摘要：本文系统研究了石榴石型荧光粉中 Cr3+ 的占位行为与近红外发射特性之间的关系。"
        "结果表明，随着掺杂浓度的提高，发射峰位发生明显红移，且发光寿命随之缩短，"
        "说明浓度猝灭效应在较高掺杂量下占据主导地位。",
        fontname=CJK_FONT,
        fontsize=9.5,
        gap_after=12.0,
    )

    b.block("1. Introduction", fontsize=H1, gap_after=4.0)
    b.block(
        "Broadband near-infrared phosphors have attracted considerable attention "
        "because they can be combined with blue light-emitting diodes to build "
        "compact and efficient NIR sources. Among the available activators, Cr3+ "
        "is particularly attractive owing to its tunable emission and strong "
        "absorption in the visible region.",
        gap_after=6.0,
    )
    b.block(
        "In garnet-type hosts, Cr3+ can occupy both octahedral and tetrahedral "
        "cation sites, and the crystal-field strength differs substantially "
        "between them. This difference is widely believed to be responsible for "
        "the unusually broad emission band. Nevertheless, a quantitative "
        "description of how the dopant concentration controls site preference "
        "is still lacking, which limits rational design of new compositions.",
        gap_after=10.0,
    )

    b.block("2. Experimental", fontsize=H1, gap_after=4.0)
    b.block("2.1 Sample preparation", fontsize=H2, gap_after=4.0)
    b.block(
        "Powder samples were synthesized by a conventional high-temperature "
        "solid-state reaction. Stoichiometric amounts of the starting oxides and "
        "carbonates were mixed thoroughly in an agate mortar, transferred to "
        "alumina crucibles, and fired in air. The Cr3+ content was varied from "
        "0.5 mol% to 8.0 mol% while keeping all other cation ratios fixed.",
        gap_after=6.0,
    )
    b.block("2.2 Characterization", fontsize=H2, gap_after=4.0)
    b.block(
        "Phase purity was examined by X-ray diffraction. Room-temperature "
        "photoluminescence spectra were recorded on a spectrofluorometer "
        "equipped with a continuous xenon lamp, and decay curves were collected "
        "using a pulsed laser diode. All measurements were performed under "
        "identical optical alignment to allow direct intensity comparison.",
        gap_after=10.0,
    )

    b.block("3. Results and Discussion", fontsize=H1, gap_after=4.0)
    b.block("3.1 Phase and microstructure", fontsize=H2, gap_after=4.0)
    b.block(
        "All diffraction patterns could be indexed to the expected garnet "
        "structure, and no impurity phase was detected up to the highest doping "
        "level. A slight contraction of the unit cell volume was observed, which "
        "is consistent with substitution of the host cation by the smaller Cr3+ "
        "ion. Scanning electron microscopy revealed well-faceted particles with "
        "an average size of approximately 4 micrometers.",
        gap_after=6.0,
    )
    b.block("3.2 Luminescence properties", fontsize=H2, gap_after=4.0)
    b.block(
        "Under blue excitation all samples exhibited a broad emission band "
        "centred in the near-infrared region. With increasing Cr3+ "
        "concentration, the emission maximum shifted from 748 nm to 812 nm, "
        "while the integrated intensity first increased and then decreased, "
        "reaching a maximum at 3.0 mol%. The decay lifetime shortened "
        "monotonically, from 128 microseconds to 41 microseconds, indicating "
        "that concentration quenching becomes dominant beyond 3.0 mol%.",
        gap_after=6.0,
    )
    b.block(
        "These trends support a model in which Cr3+ preferentially occupies the "
        "octahedral site at low concentration and increasingly populates the "
        "tetrahedral site as the dopant content rises. Because the tetrahedral "
        "crystal field is weaker, emission from that site is redshifted, which "
        "explains the observed shift of the band maximum.",
        gap_after=10.0,
    )

    b.block("4. Conclusions", fontsize=H1, gap_after=4.0)
    b.block(
        "We have shown that the near-infrared emission of garnet-type phosphors "
        "is governed by the site distribution of Cr3+. A doping level of "
        "3.0 mol% provides the best compromise between absorption strength and "
        "quenching, and the corresponding emission maximum lies at 781 nm. "
        "These findings offer a practical guideline for designing broadband NIR "
        "phosphors with tailored emission profiles.",
        gap_after=12.0,
    )

    b.block("References", fontsize=H1, gap_after=4.0)
    references = [
        "[1] A. Demo, B. Example, Broadband NIR phosphors for compact light "
        "sources, J. Lumin. 240 (2021) 118401.",
        "[2] C. Sample, D. Placeholder, Site engineering in garnet hosts, "
        "Chem. Mater. 33 (2021) 5521-5530.",
        "[3] E. Test, F. Fixture, Concentration quenching in Cr3+-doped oxides, "
        "Phys. Rev. B 104 (2021) 144102.",
        "[4] G. Mock, H. Dummy, Crystal-field effects on Cr3+ emission, "
        "Opt. Mater. 121 (2021) 111550.",
        "[5] I. Alpha, J. Beta, High-temperature synthesis of garnet phosphors, "
        "J. Am. Ceram. Soc. 105 (2022) 3312-3321.",
    ]
    for ref in references:
        b.block(ref, fontsize=8.5, gap_after=3.0)

    return b.save(Path(__file__).resolve().parents[1] / "samples" / "nir-phosphor-demo.pdf")


def main() -> int:
    path = build()
    size_kb = path.stat().st_size / 1024
    print(f"已生成样本 PDF：{path}（{size_kb:.1f} KB）")
    with pymupdf.open(str(path)) as doc:
        print(f"页数：{doc.page_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
