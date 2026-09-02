"""期刊交付约束注册表（P3）：每个档一份有官方出处的约束值。

设计边界（对比规格 §3）：
- preset（cn/nature）管语言、字体族与纹理；``journal=`` 只覆盖**有官方
  出处**的交付约束（栏宽、字号区间、图高上限）。
- 官方没有明文的约束**不造数**：IEEE/PNAS 官方页面均无背景网格禁令、
  彩色文字禁令与线宽区间 → 这三类的档级硬拒只属于 nature
  （grid_allowed/colored_text_allowed=True、max_line_pt=None 即"官方
  无此约束，run_qa 不设档级检查"）。
- nature/cn 也注册为约束档：journal=None 时约束值取当前 preset 的档，
  数值与迁移前的 qa.py 硬编码逐项相等（tests/test_journal_profiles.py
  钉死）——qa.py 里不再有第二真值源。
- profile 不可变（frozen dataclass）。``reviewed_on`` 是人工核对官方
  页面的日期；官方改版后须重核并更新该日期。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class JournalProfile:
    """一份期刊（或 preset 自带）的交付约束集。

    kind: "journal"（官方期刊档）| "preset-native"（cn/nature 档自带，
    数值源自本库既有规范而非期刊官方页面）。
    sources: (约束项, 官方 URL) 逐项出处；source_url 是主出处。
    notes: (约束项, 说明)，记录"官方无明文"与推导口径。
    """
    name: str
    display: str                     # QA 消息里的档名（Nature / IEEE / …）
    kind: str
    source_url: str | None
    reviewed_on: str
    column_widths_mm: dict           # width 键 → mm（仅收官方定义的键）
    max_height_mm: float
    base_font_pt: float | None       # None = 不覆盖 preset 基准字号
    min_font_pt: float | None        # None = 该档无字号下限检查
    max_font_pt: float | None        # None = 该档无字号上限检查
    max_line_pt: float | None        # None = 该档无线宽上限检查
    line_range: str | None           # 消息里引用的官方区间原文，如 "0.25–1pt"
    grid_allowed: bool
    colored_text_allowed: bool
    panel_label_style: str | None    # 面板标签样式，如 "a" / "(a)"
    panel_label_size: float | None   # 面板标签字号（高于上限的档豁免 a–h）
    sources: tuple = field(default=())
    notes: tuple = field(default=())


_J_IEEE = ("https://journals.ieeeauthorcenter.ieee.org/create-your-"
           "ieee-journal-article/create-graphics-for-your-article/"
           "resolution-and-size/")
_J_IEEE_CONF = ("https://conferences.ieeeauthorcenter.ieee.org/"
                "write-your-paper/improve-your-graphics/")
_J_IEEE_PES = ("https://ieee-pes.org/publications/authors-kit/"
               "preparation-of-a-formatted-technical-work/")
_J_PNAS = "https://www.pnas.org/author-center/submitting-your-manuscript"
_J_NATURE = "https://www.nature.com/nature/for-authors/final-submission"

_REVIEWED = "2026-09-01"

JOURNALS = {
    # ---- preset 自带约束档：数值 = qa.py 迁移前的硬编码（非期刊官方出处）
    "nature": JournalProfile(
        name="nature", display="Nature", kind="preset-native",
        source_url=_J_NATURE, reviewed_on=_REVIEWED,
        column_widths_mm={"single": 89, "onehalf": 136, "double": 183,
                          "cn": 150},
        max_height_mm=170.0,
        base_font_pt=7.0, min_font_pt=5.0, max_font_pt=7.0,
        max_line_pt=1.0, line_range="0.25–1pt",
        grid_allowed=False, colored_text_allowed=False,
        panel_label_style="a", panel_label_size=8.0,
        notes=(
            ("column_widths_mm", "Nature 系栏宽常量（core/style.py "
             "COLUMN_WIDTHS），kind=preset-native：沿用本库既有规范值"),
            ("max_height_mm", "Nature 明文 170mm（整页图含图注可用高度），"
             "与 core/style.py MAX_HEIGHT_MM 同源"),
            ("source_url", "Nature 投稿指南图件章节；本档数值在 P3 前已是"
             "库内硬编码，此处仅登记出处，未改任何值"),
        ),
    ),
    "cn": JournalProfile(
        name="cn", display="cn", kind="preset-native",
        source_url=None, reviewed_on=_REVIEWED,
        column_widths_mm={"single": 89, "onehalf": 136, "double": 183,
                          "cn": 150},
        max_height_mm=170.0,
        base_font_pt=9.0, min_font_pt=6.5, max_font_pt=None,
        max_line_pt=None, line_range=None,
        grid_allowed=True, colored_text_allowed=True,
        panel_label_style="(a)", panel_label_size=10.0,
        notes=(
            ("source_url", "非期刊档：中文数模 A4 交付约定（本库 SKILL "
             "规范），无官方期刊页面"),
            ("max_height_mm", "沿用 Nature 上限 170mm（本库既有行为）"),
        ),
    ),
    # ---- 新增期刊档：只收官方页面明文给出的值
    "ieee": JournalProfile(
        name="ieee", display="IEEE", kind="journal",
        source_url=_J_IEEE, reviewed_on=_REVIEWED,
        column_widths_mm={"single": 88.9, "double": 182.0},
        max_height_mm=220.0,
        base_font_pt=9.0, min_font_pt=8.0, max_font_pt=10.0,
        max_line_pt=None, line_range=None,
        grid_allowed=True, colored_text_allowed=True,
        panel_label_style="(a)", panel_label_size=None,
        sources=(
            ("column_widths_mm", _J_IEEE),
            ("max_height_mm", _J_IEEE_CONF),
            ("base_font_pt", _J_IEEE_CONF),
            ("min_font_pt", _J_IEEE_PES),
            ("max_font_pt", _J_IEEE_CONF),
        ),
        notes=(
            ("column_widths_mm", "官方原文：'One column width: 3.5 inches, "
             "88.9 millimeters, or 21 picas'；'Two columns width: 7.16 "
             "inches, 182 millimeters, or 43 picas'。IEEE 未定义 1.5 栏"),
            ("max_height_mm", "官方原文：'no larger than 7.16 x 8.8 inches, "
             "182 x 220 millimeters'——同页 in/mm 表述取其 mm 值"),
            ("base_font_pt", "官方原文：'Type should appear approximately "
             "9-10 point when viewed at full size'，取 9pt 为档内基准"),
            ("min_font_pt", "IEEE PES 作者套件原文：'Figure labels should "
             "be legible, approximately 8- to 10-point type'，下沿 8pt"),
            ("max_font_pt", "同上 '9-10 point' 推荐区间取上沿 10pt 为档内"
             "上限（官方为推荐口径，非硬性禁令）"),
            ("grid_allowed", "官方页面无背景网格禁令——不造数"),
            ("colored_text_allowed", "官方页面无彩色文字禁令（可达性建议"
             "色彩+形状冗余编码）——不造数"),
            ("max_line_pt", "官方页面无线宽区间——不造数"),
            ("panel_label_style", "IEEE 模板子图约定 (a)(b)，非官方硬性"
             "条款，run_qa 不据此检查"),
        ),
    ),
    "pnas": JournalProfile(
        name="pnas", display="PNAS", kind="journal",
        source_url=_J_PNAS, reviewed_on=_REVIEWED,
        column_widths_mm={"single": 90.0, "onehalf": 110.0,
                          "double": 180.0},
        max_height_mm=220.0,
        base_font_pt=None, min_font_pt=6.0, max_font_pt=12.0,
        max_line_pt=None, line_range=None,
        grid_allowed=True, colored_text_allowed=True,
        panel_label_style=None, panel_label_size=None,
        sources=(
            ("column_widths_mm", _J_PNAS),
            ("max_height_mm", _J_PNAS),
            ("min_font_pt", _J_PNAS),
            ("max_font_pt", _J_PNAS),
        ),
        notes=(
            ("column_widths_mm", "官方原文：'Small: approximately 9 cm x "
             "6 cm / Medium: approximately 11 cm x 11 cm / Large: "
             "approximately 18 cm x 22 cm'，映射 single/onehalf/double"),
            ("max_height_mm", "取官方 Large 档高 22 cm"),
            ("min_font_pt", "官方原文：'Ensure that all numbers, letters, "
             "and symbols are no smaller than 6 points (2 mm) and no "
             "larger than 12 points (6 mm) after reduction.'——图按 final "
             "size 交付，即终稿字号硬区间"),
            ("base_font_pt", "官方只给 6-12pt 硬区间、未给基准字号，取 "
             "None = 沿用 preset 基准（9pt/7pt 均在区间内）"),
            ("grid_allowed", "官方页面无背景网格禁令——不造数"),
            ("colored_text_allowed", "官方页面无彩色文字禁令——不造数"),
            ("max_line_pt", "官方页面无线宽区间——不造数"),
            ("panel_label_style", "官方页面未给面板标签样式——不造数"),
        ),
    ),
}


def get_journal(name: str) -> JournalProfile:
    """按名取档；未知档报出全部合法档名并指到本模块与官方出处。"""
    if name not in JOURNALS:
        legal = sorted(JOURNALS)
        raise ValueError(
            f"未知期刊档 {name!r}，可选：{legal}"
            f"——各档约束值与官方出处见 core/journals.py"
            f"（get_journal(name).sources / source_url）")
    return JOURNALS[name]


def journal_names() -> list[str]:
    return sorted(JOURNALS)
