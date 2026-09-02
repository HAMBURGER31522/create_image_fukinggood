"""P3 期刊档注册表：ieee / pnas 约束档 + ``journal=`` 正交关键字。

两侧用例约定同 test_qa_checks：每条新行为都配「该触发」与「不该触发」
两枚用例。另设一组 **journal=None 快照**用例把 cn/nature 在 P3 之前的
rcParams / 栏宽 / QA 行为逐项钉死——P3 必须是纯新增，任何一条快照变红
都说明得罪了现有档位语义。

约束值出处（官方页面，复核日期 2026-09-01，机器可读出处也在
core/journals.py 各 profile 的 sources 里）：
- IEEE 栏宽：IEEE Author Center Journals "Resolution and Size"
  （one column 3.5 in / 88.9 mm；two columns 7.16 in / 182 mm）。
- IEEE 字号与整图上限：IEEE Author Center Conferences "Improve Your
  Graphics"（type ~9-10 pt；no larger than 7.16 x 8.8 in / 182 x 220 mm）
  与 IEEE PES 作者套件（"Figure labels should be legible, approximately
  8- to 10-point type"）。
- PNAS 尺寸与字号：pnas.org Author Center "Submitting Your Manuscript"
  （Small ~9x6 cm / Medium ~11x11 cm / Large ~18x22 cm；字号 6-12 pt
  硬区间）。
"""
import re
import sys
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from core import (apply_style, new_figure, run_qa, save_figure,       # noqa: E402
                  current_preset, current_journal, get_journal,
                  COLUMN_WIDTHS, ptx)
from core.journals import JOURNALS, JournalProfile                    # noqa: E402


@pytest.fixture(autouse=True)
def _styled():
    apply_style()
    yield
    plt.close("all")


def probs(fig, **kw):
    return run_qa(fig, strict=False, **kw)


def hit(fig, fragment, **kw):
    return any(fragment in p for p in probs(fig, **kw))


def _clean_fig(width="single"):
    """最小但能全绿通过 run_qa 的图（构造同 test_manifest._clean_fig）。

    散点要够密：双栏（182mm）面板面积大，40 个点会触发 sparse_panel
    墨迹下限，这里固定 200 点保证各栏宽下都过。
    """
    rng = np.random.default_rng(7)
    fig, ax = new_figure(width)
    ax.scatter(rng.normal(size=200), rng.normal(size=200), s=18)
    ax.set_title("两组分布差异可见")
    ax.text(0.05, 0.95, "样本充分", transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", fc="white", ec="0.4"))
    run_qa(fig)          # strict：夹具图若退化，这里直接炸
    return fig


def _pdf_width_mm(path):
    raw = pathlib.Path(path).read_bytes()
    m = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", raw)
    assert m, "PDF 里找不到 MediaBox，无法实测落盘宽度"
    return float(m.group(1)) * 25.4 / 72


# === 0. journal=None：cn/nature 既有行为快照（P3 纯新增的锚） ============

def test_journal_none_rcparams_unchanged_cn():
    apply_style("cn")
    assert plt.rcParams["font.size"] == 9.0
    assert plt.rcParams["figure.titlesize"] == 10.5
    assert plt.rcParams["axes.titlesize"] == 10.5
    assert plt.rcParams["axes.grid"] is True
    assert plt.rcParams["grid.alpha"] == 0.22
    assert plt.rcParams["lines.linewidth"] == 1.2
    assert plt.rcParams["axes.linewidth"] == 0.6
    assert plt.rcParams["legend.fontsize"] == 7.5
    assert plt.rcParams["mathtext.fontset"] == "stix"


def test_journal_none_rcparams_unchanged_nature():
    apply_style("nature")
    assert plt.rcParams["font.size"] == 7.0
    assert plt.rcParams["figure.titlesize"] == 7.0
    assert plt.rcParams["axes.grid"] is False
    assert plt.rcParams["lines.linewidth"] == 1.0
    assert plt.rcParams["axes.linewidth"] == 0.5
    assert plt.rcParams["legend.fontsize"] == 5.5
    assert plt.rcParams["mathtext.fontset"] == "dejavusans"


def test_journal_none_column_widths_unchanged():
    for preset in ("cn", "nature"):
        apply_style(preset)
        assert COLUMN_WIDTHS == {"single": 89, "onehalf": 136,
                                 "double": 183, "cn": 150}
        for key, mm in COLUMN_WIDTHS.items():
            fig, _ = new_figure(key)
            assert fig.get_figwidth() * 25.4 == pytest.approx(mm)


def test_journal_none_qa_unchanged_nature_rejections():
    """nature 档禁网格/彩色文字/粗线/超 7pt：P3 后仍必须逐项拒绝。"""
    apply_style("nature")
    fig, ax = new_figure("single")
    ax.plot([0, 1], [0, 1], linewidth=3.0)
    ax.grid(True)
    ax.text(0.2, 0.6, "标记", color="tab:blue")
    ax.text(0.2, 0.3, "标题", fontsize=7.5)
    assert hit(fig, "背景网格")
    assert hit(fig, "彩色文字")
    assert hit(fig, "线宽 > 1pt")
    assert hit(fig, "超 Nature 正文字号上限")


def test_journal_none_ptx_unchanged():
    apply_style("nature")
    assert ptx(3.0, "lw") == 1.0            # nature 档封顶 1pt
    assert ptx(20.0, "font") == 7.0         # 夹到 7pt 上限
    assert ptx(2.0, "font") == 5.0          # 夹到 5pt 下限
    apply_style("cn")
    assert ptx(3.0, "lw") == 3.0            # cn 无线宽上限
    assert ptx(20.0, "font") == 20.0        # cn 无字号上限
    assert ptx(2.0, "font") == 6.5          # cn 下限 6.5


# === 1. 注册表与官方出处 ================================================

def test_registry_exposes_exactly_four_profiles():
    assert set(JOURNALS) == {"nature", "cn", "ieee", "pnas"}


def test_ieee_profile_values_match_official_specs():
    p = get_journal("ieee")
    assert p.column_widths_mm == {"single": 88.9, "double": 182.0}
    assert p.base_font_pt == 9.0
    assert p.min_font_pt == 8.0
    assert p.max_font_pt == 10.0
    assert p.max_height_mm == 220.0
    # 官方未设禁令的项不造数：无网格/彩色文字禁令，无线宽区间
    assert p.grid_allowed is True
    assert p.colored_text_allowed is True
    assert p.max_line_pt is None
    assert p.source_url.startswith("https://")
    assert "ieee" in p.source_url
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.reviewed_on)
    assert p.sources and all(u.startswith("https://")
                             for _, u in p.sources)


def test_pnas_profile_values_match_official_specs():
    p = get_journal("pnas")
    assert p.column_widths_mm == {"single": 90.0, "onehalf": 110.0,
                                  "double": 180.0}
    assert p.min_font_pt == 6.0
    assert p.max_font_pt == 12.0
    assert p.base_font_pt is None          # 官方只给 6-12pt 区间，未给基准
    assert p.max_height_mm == 220.0
    assert p.grid_allowed is True
    assert p.colored_text_allowed is True
    assert p.max_line_pt is None
    assert p.source_url.startswith("https://")
    assert "pnas" in p.source_url
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.reviewed_on)


def test_preset_native_profiles_carry_legacy_values():
    """nature/cn 约束档必须逐项等于第 16 轮前的硬编码（单一真值源迁移）。"""
    nat, cn = get_journal("nature"), get_journal("cn")
    assert (nat.min_font_pt, nat.max_font_pt, nat.max_line_pt) == \
        (5.0, 7.0, 1.0)
    assert nat.max_height_mm == 170.0
    assert nat.grid_allowed is False and nat.colored_text_allowed is False
    assert cn.min_font_pt == 6.5 and cn.max_font_pt is None
    assert cn.max_line_pt is None
    assert cn.grid_allowed is True and cn.colored_text_allowed is True
    for prof in (nat, cn):
        assert prof.column_widths_mm == COLUMN_WIDTHS


def test_unknown_journal_raises_with_legal_names_and_source_doc():
    with pytest.raises(ValueError) as ei:
        apply_style("cn", journal="sci")
    msg = str(ei.value)
    assert "ieee" in msg and "pnas" in msg
    assert "journals" in msg or "http" in msg   # 指到来源文档/出处
    with pytest.raises(ValueError):
        get_journal("nature-journal")


def test_profiles_are_immutable():
    p = get_journal("ieee")
    with pytest.raises(Exception):
        p.min_font_pt = 5.0


# === 2. apply_style 集成 ================================================

def test_apply_style_accepts_journal_on_both_presets():
    for preset in ("cn", "nature"):
        for journal in ("ieee", "pnas"):
            apply_style(preset, journal=journal)
            assert current_preset() == preset
            assert current_journal() == journal


def test_journal_with_base_font_overrides_preset_base():
    apply_style("nature", journal="ieee")
    assert plt.rcParams["font.size"] == 9.0   # IEEE 官方 ~9-10pt


def test_journal_without_base_keeps_preset_base():
    apply_style("cn", journal="pnas")
    assert plt.rcParams["font.size"] == 9.0   # pnas 未给基准 → 沿用 cn 9
    apply_style("nature", journal="pnas")
    assert plt.rcParams["font.size"] == 7.0


def test_journal_title_and_derived_sizes_stay_inside_journal_range():
    """cn 纹理 + ieee 档：图题 10.5 会被夹进 IEEE 8-10pt，图例 ≥8pt。"""
    apply_style("cn", journal="ieee")
    assert plt.rcParams["figure.titlesize"] == pytest.approx(10.0)
    assert plt.rcParams["axes.titlesize"] == pytest.approx(10.0)
    assert plt.rcParams["legend.fontsize"] >= 8.0
    assert plt.rcParams["xtick.labelsize"] >= 8.0


def test_reread_apply_style_resets_journal():
    apply_style("cn", journal="ieee")
    apply_style("cn")
    assert current_journal() is None
    assert plt.rcParams["font.size"] == 9.0


# === 3. new_figure 栏宽 =================================================

def test_new_figure_uses_journal_widths():
    apply_style("cn", journal="ieee")
    fig, _ = new_figure("single")
    assert fig.get_figwidth() * 25.4 == pytest.approx(88.9, abs=1e-6)
    fig, _ = new_figure("double")
    assert fig.get_figwidth() * 25.4 == pytest.approx(182.0, abs=1e-6)
    apply_style("cn", journal="pnas")
    for key, mm in (("single", 90.0), ("onehalf", 110.0), ("double", 180.0)):
        fig, _ = new_figure(key)
        assert fig.get_figwidth() * 25.4 == pytest.approx(mm, abs=1e-6)


def test_new_figure_rejects_width_journal_does_not_define():
    apply_style("cn", journal="ieee")          # IEEE 无 1.5 栏宽
    with pytest.raises(ValueError) as ei:
        new_figure("onehalf")
    assert "single" in str(ei.value) and "double" in str(ei.value)


def test_new_figure_numeric_width_passthrough_under_journal():
    apply_style("cn", journal="ieee")
    fig, _ = new_figure(150.0)                 # 显式 mm 数值不受档名限制
    assert fig.get_figwidth() * 25.4 == pytest.approx(150.0)


# === 4. run_qa 期刊约束 =================================================

def test_journal_min_font_rejects_below_floor():
    apply_style("cn", journal="pnas")
    fig, ax = new_figure("single")
    ax.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    ax.text(0.1, 0.5, "过小", fontsize=4.0)
    assert hit(fig, "字号 < 6.0pt")
    apply_style("cn", journal="ieee")
    fig, ax = new_figure("single")
    ax.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    ax.text(0.1, 0.5, "过小", fontsize=7.5)
    assert hit(fig, "字号 < 8.0pt")


def test_journal_min_font_allows_between_preset_and_journal_floor():
    """6.2pt 在 cn 档(≥6.5)必拒、在 pnas 档(≥6)合法——钉住按档取值。"""
    apply_style("cn", journal="pnas")
    fig, ax = new_figure("single")
    ax.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    ax.text(0.1, 0.5, "六点二", fontsize=6.2)
    assert not hit(fig, "字号 <")


def test_journal_max_font_rejects_above_ceiling():
    apply_style("cn", journal="pnas")
    fig, ax = new_figure("single")
    ax.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    ax.text(0.1, 0.5, "过大", fontsize=13.0)
    assert hit(fig, "超 PNAS 正文字号上限")
    apply_style("cn", journal="ieee")
    fig, ax = new_figure("single")
    ax.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    ax.text(0.1, 0.5, "过大", fontsize=10.5)
    assert hit(fig, "超 IEEE 正文字号上限")


def test_journal_max_font_allows_within_ceiling():
    apply_style("cn", journal="pnas")
    fig, ax = new_figure("single")
    ax.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    ax.text(0.1, 0.5, "十一点", fontsize=11.0)
    assert not hit(fig, "正文字号上限")


def test_journal_qa_accepts_journal_width_and_rejects_off_width():
    apply_style("cn", journal="pnas")
    fig = _clean_fig("onehalf")                # 110mm：pnas 1.5 栏合法
    assert not hit(fig, "交付宽")
    fig, ax = new_figure(160.0)                # 160mm 不在 90/110/180 ±3
    ax.scatter(np.random.default_rng(7).normal(size=40),
               np.random.default_rng(7).normal(size=40), s=18)
    ax.set_title("两组分布差异可见")
    ax.text(0.05, 0.95, "样本充分", transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", fc="white", ec="0.4"))
    assert hit(fig, "交付宽")


def test_journal_height_cap():
    apply_style("cn", journal="pnas")
    fig, ax = new_figure(90.0, ratio=210 / 90)   # 210mm：pnas 上限内合法
    ax.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    assert not hit(fig, "超上限")
    fig, ax = new_figure(90.0, ratio=230 / 90)   # 230mm：超 220mm
    ax.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    assert hit(fig, "超上限 220mm")
    apply_style("cn")                            # journal=None 仍 170mm
    fig, ax = new_figure(90.0, ratio=180 / 90)
    ax.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    assert hit(fig, "超上限 170mm")


def test_journal_allows_grid_colored_text_and_thick_lines():
    """IEEE/PNAS 官方无网格/彩色文字/线宽禁令 → 档级检查不造数。"""
    apply_style("cn", journal="pnas")
    fig, ax = new_figure("onehalf")
    ax.plot([0, 1], [0, 1], linewidth=3.0)
    ax.grid(True)
    ax.text(0.2, 0.6, "标记", color="tab:blue")
    assert not hit(fig, "背景网格")
    assert not hit(fig, "彩色文字")
    assert not hit(fig, "线宽 >")


# === 5. save_figure 门禁与落盘宽度 ======================================

@pytest.mark.parametrize("journal,width_key,target_mm", [
    ("ieee", "single", 88.9),
    ("ieee", "double", 182.0),
    ("pnas", "single", 90.0),
    ("pnas", "onehalf", 110.0),
])
def test_journal_pdf_delivered_width_matches_profile(tmp_path, journal,
                                                     width_key, target_mm):
    apply_style("cn", journal=journal)
    fig = _clean_fig(width_key)                # 内部先 run_qa(strict)
    out = save_figure(fig, str(tmp_path / "stem"), formats=("pdf",))
    assert len(out) == 1
    assert abs(_pdf_width_mm(out[0]) - target_mm) <= 3.0


def test_failed_qa_writes_zero_files_under_journal(tmp_path):
    apply_style("cn", journal="pnas")
    fig, ax = new_figure("single")
    ax.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    ax.text(0.1, 0.5, "过小", fontsize=4.0)
    with pytest.raises(AssertionError):
        run_qa(fig, strict=True)
    with pytest.raises(RuntimeError):
        save_figure(fig, str(tmp_path / "stem"), formats=("pdf",))
    assert list(tmp_path.iterdir()) == []      # 0 文件落盘


# === 6. ptx 与档联动 ====================================================

def test_ptx_clamps_into_journal_font_range():
    apply_style("cn", journal="pnas")
    assert ptx(20.0, "font") == pytest.approx(12.0)
    assert ptx(4.0, "font") == pytest.approx(6.0)


def test_ptx_lw_uncapped_when_journal_has_no_line_limit():
    apply_style("cn", journal="pnas")
    assert ptx(3.0, "lw") == 3.0
