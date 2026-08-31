"""qa.run_qa 新增/升级检查的回归测试。

每条检查配一个「该触发」与一个「不该触发」的最小用例——只有两侧都
钉住，检查才既有检出力又不误伤。断言一律走 strict=False 返回的问题
列表并匹配消息片段，这样其他无关检查的噪声不会干扰判定。
"""
import sys
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from core import apply_style, new_figure, run_qa, PALETTE  # noqa: E402
from core.layout import small_multiples                     # noqa: E402


@pytest.fixture(autouse=True)
def _styled():
    apply_style()
    yield
    plt.close("all")


def probs(fig, **kw):
    return run_qa(fig, strict=False, **kw)


def hit(fig, fragment, **kw):
    return any(fragment in p for p in probs(fig, **kw))


# --- C1 图题与图内注释的同名量数值矛盾 ---------------------------------

def test_suptitle_and_statbox_disagree_on_same_quantity():
    fig, ax = new_figure("onehalf")
    ax.plot([1, 2, 3], [1, 2, 3])
    fig.suptitle("迭代收敛：RMS = 5.07 cm")
    ax.text(0.1, 0.8, "RMS = 6.20 cm", transform=ax.transAxes,
            bbox=dict(boxstyle="round", fc="w"))
    assert hit(fig, "两个不同数值")


def test_same_quantity_at_different_precision_is_not_flagged():
    fig, ax = new_figure("onehalf")
    ax.plot([1, 2, 3], [1, 2, 3])
    fig.suptitle("迭代收敛：RMS = 5.1 cm")
    ax.text(0.1, 0.8, "RMS = 5.07 cm", transform=ax.transAxes,
            bbox=dict(boxstyle="round", fc="w"))
    assert not hit(fig, "两个不同数值")


def test_per_panel_sample_size_may_differ_across_panels():
    fig, axes = plt.subplots(1, 2)
    for a, n in zip(axes, (30, 45)):
        a.plot([1, 2, 3], [1, 2, 3])
        a.text(0.1, 0.8, f"n = {n}", transform=a.transAxes,
               bbox=dict(boxstyle="round", fc="w"))
    assert not hit(fig, "两个不同数值")


# --- C2 可达性：无冗余编码时升为硬错 -----------------------------------

def _six_color_fig(markers):
    fig, ax = new_figure("onehalf")
    x = np.linspace(0, 1, 20)
    for i, c in enumerate(PALETTE[:6]):
        ax.plot(x, x + i, color=c,
                marker=markers[i] if markers else None)
    return fig


def test_six_categorical_colors_without_redundancy_is_hard_error():
    assert hit(_six_color_fig(None), "可达性")


def test_six_categorical_colors_with_marker_redundancy_stays_warning():
    assert not hit(_six_color_fig(["o", "s", "^", "D", "v", "P"]),
                   "可达性")


def test_accessibility_is_waivable():
    assert not hit(_six_color_fig(None), "可达性",
                   allow=("accessibility",))


# --- C3 填充带无任何说明：升为硬错 -------------------------------------

def test_fill_band_without_legend_or_label_is_hard_error():
    fig, ax = new_figure("onehalf")
    x = np.linspace(0, 1, 20)
    ax.plot(x, x)
    ax.fill_between(x, x - 0.1, x + 0.1, alpha=0.2)
    assert hit(fig, "无法得知其含义")


def test_fill_band_with_legend_is_not_flagged():
    fig, ax = new_figure("onehalf")
    x = np.linspace(0, 1, 20)
    ax.plot(x, x, label="均值")
    ax.fill_between(x, x - 0.1, x + 0.1, alpha=0.2, label="95% CI")
    ax.legend()
    assert not hit(fig, "无法得知其含义")


# --- C4 跨面板重复曲线 -------------------------------------------------

def test_identical_series_drawn_in_two_panels_is_flagged():
    fig, axes = plt.subplots(1, 2)
    x = np.linspace(0, 1, 40)
    y = np.sin(x * 6)
    for a in axes:
        a.plot(x, y)
    assert hit(fig, "重复画了同一条数据系列")


def test_different_series_in_two_panels_is_not_flagged():
    fig, axes = plt.subplots(1, 2)
    x = np.linspace(0, 1, 40)
    axes[0].plot(x, np.sin(x * 6))
    axes[1].plot(x, np.cos(x * 6))
    assert not hit(fig, "重复画了同一条数据系列")


def test_shared_baseline_reference_line_is_not_flagged():
    fig, axes = plt.subplots(1, 2)
    x = np.linspace(0, 1, 40)
    axes[0].plot(x, np.sin(x * 6))
    axes[1].plot(x, np.cos(x * 6))
    for a in axes:
        a.axhline(0.0, ls="--", lw=0.8)
    assert not hit(fig, "重复画了同一条数据系列")


# --- C5 小倍数面板色标范围不一致 ---------------------------------------

def _sm_fig(clims):
    fig, axes = small_multiples(2, ncols=2)
    field = np.random.default_rng(0).random((8, 8))
    for a, (lo, hi) in zip(axes, clims):
        a.imshow(field, cmap="viridis", vmin=lo, vmax=hi)
    return fig


def test_small_multiples_with_mismatched_clim_is_flagged():
    assert hit(_sm_fig([(0, 1), (0, 2)]), "色标范围不一致")


def test_small_multiples_with_shared_clim_is_not_flagged():
    assert not hit(_sm_fig([(0, 1), (0, 1)]), "色标范围不一致")


# --- 回归：run_all 暴露的两类误伤 --------------------------------------

def test_grayscale_only_gap_stays_warning():
    """灰度亮度差不足不该阻断落盘。

    SKILL.md §3d 给的判据是**色盲色距**（N=2→0.93 … N=6→0.05），
    灰度可分性从未被声明为硬门槛。Okabe-Ito 的橙(#d55e00)与蓝(#0072b2)
    色距安全但灰度亮度接近——把它判成硬错会顶掉 cluster_scatter /
    route_map / composition 等一批构图正当的图。
    """
    fig, ax = new_figure("onehalf")
    x = np.linspace(0, 1, 20)
    ax.plot(x, x, color="#d55e00")
    ax.plot(x, x + 1, color="#0072b2")
    assert not hit(fig, "可达性")


def test_shared_gray_reference_curve_across_panels_is_not_flagged():
    """各面板重复画同一条中性色参考线是正当的构图，不是重复信息。

    small_multiples_frames 每帧都画同一个灰色口径圆当面板边界——那是
    刻度不是数据。判据取"有没有上分类色"，与可达性检查里"中性灰=基线色，
    不参与分类"的口径一致。
    """
    fig, axes = plt.subplots(1, 2)
    th = np.linspace(0, 2 * np.pi, 100)
    for a, ph in zip(axes, (0.0, 1.0)):
        a.plot(np.linspace(0, 1, 40), np.sin(np.linspace(0, 1, 40) * 6 + ph),
               color=PALETTE[0])
        a.plot(np.cos(th), np.sin(th), color="0.3", ls="--", lw=0.7)
    assert not hit(fig, "重复画了同一条数据系列")


# --- 稀疏数据构图（q2_1 / q4_1 暴露的一组缺陷）------------------------

def _overlapping_titles_fig():
    """把图题精确压到面板标题上——q2_1/q4_1 实测缺陷的最小复现。"""
    fig, ax = new_figure("onehalf")
    ax.plot([1, 2, 3], [1, 2, 3])
    ax.set_title("面板标题")
    fig.canvas.draw()
    bb = ax.title.get_window_extent(fig.canvas.get_renderer())
    y = fig.transFigure.inverted().transform((0, bb.y0 + bb.height / 2))[1]
    fig.suptitle("结论句：某某指标仅达标一档，三项复核一致", y=y)
    return fig


def test_suptitle_overlapping_panel_title_is_hard_error():
    assert hit(_overlapping_titles_fig(), "图题与面板标题重叠")


def test_normal_suptitle_above_panel_title_is_not_flagged():
    fig, ax = new_figure("onehalf")
    ax.plot([1, 2, 3], [1, 2, 3])
    ax.set_title("面板标题")
    fig.suptitle("结论句")
    fig.tight_layout()
    assert not hit(fig, "图题与面板标题重叠")


def test_four_discrete_points_joined_by_a_line_is_hard_error():
    """4 个离散档位连折线：折线宣称档位之间可插值，语义就错了。"""
    fig, ax = new_figure("onehalf")
    ax.plot([0.5, 0.6, 0.7, 1.0], [0.081, 0.216, 0.497, 0.9935],
            "o-", color=PALETTE[0])
    assert hit(fig, "个点用折线连起来")


def test_slopegraph_two_point_lines_are_not_flagged():
    """斜率图就是每条线两个点，是正当构图。"""
    fig, ax = new_figure("onehalf")
    for i in range(6):
        ax.plot([0, 1], [i, i * 0.6 + 1], "o-", color=PALETTE[i % 6])
    assert not hit(fig, "个点用折线连起来")


def test_dense_curve_is_not_flagged():
    fig, ax = new_figure("onehalf")
    x = np.linspace(0, 1, 60)
    ax.plot(x, np.sin(x * 6), "-", color=PALETTE[0])
    assert not hit(fig, "个点用折线连起来")


def test_outlier_stretched_axis_is_hard_error():
    """一个离群点把 y 轴撑到 14000，其余 4 个点压成一条线（q4_1 面板 d）。"""
    fig, ax = new_figure("onehalf")
    ax.scatter([2.6, 5, 10, 20], [0, 0, 0, 0], color=PALETTE[0])
    ax.scatter([40], [3300], color=PALETTE[1])
    ax.set_ylim(0, 14000)
    assert hit(fig, "被压成一条线")


def test_axis_using_most_of_its_span_is_not_flagged():
    fig, ax = new_figure("onehalf")
    ax.scatter([1, 2, 3, 4], [0.1, 0.5, 0.8, 0.95], color=PALETTE[0])
    ax.set_ylim(0, 1)
    assert not hit(fig, "被压成一条线")


def test_probability_axis_beyond_zero_one_is_hard_error():
    """概率轴放到 −0.25～1.12：负概率没有意义（q2_1 面板 a）。"""
    fig, ax = new_figure("onehalf")
    ax.plot([1, 2, 3, 4], [0.081, 0.216, 0.497, 0.9935],
            "o", color=PALETTE[0])
    ax.set_ylabel("导通概率 P")      # 判据要求轴标题确实说了是概率
    ax.set_ylim(-0.25, 1.12)
    assert hit(fig, "比例/概率轴超出")


def test_probability_axis_clamped_to_zero_one_is_not_flagged():
    fig, ax = new_figure("onehalf")
    ax.plot([1, 2, 3, 4], [0.081, 0.216, 0.497, 0.9935],
            "o", color=PALETTE[0])
    ax.set_ylim(0, 1)
    assert not hit(fig, "比例/概率轴超出")


def test_non_proportional_data_is_not_treated_as_probability_axis():
    fig, ax = new_figure("onehalf")
    ax.plot([1, 2, 3, 4], [120, 340, 780, 1500], "o", color=PALETTE[0])
    ax.set_ylim(-200, 1800)
    assert not hit(fig, "比例/概率轴超出")


# --- 回归：run_all 暴露的第二批误伤 ------------------------------------

def test_parallel_coordinates_lines_are_not_flagged():
    """平行坐标每条线连 N 个维度轴，连线正是它的构图本体。

    判据取"这样的稀疏连线有几条"：平行坐标是一整束（composition.py
    实测 5 维多条），q2_1 那种病灶只有孤零零一条。
    """
    fig, ax = new_figure("onehalf")
    rng = np.random.default_rng(3)
    for i in range(10):
        ax.plot(range(5), rng.random(5), "-", color=PALETTE[i % 6])
    assert not hit(fig, "个点用折线连起来")


def test_slopegraph_category_axis_widened_for_labels_is_not_flagged():
    """斜率图的 x 是位置类别（只有 0 和 1 两个取值），不是比例轴。

    轴限放宽到 −0.45～1.30 是为了两端直标留位，是正确构图。
    比例轴的判据因此要求轴上至少有 4 个不同取值。
    """
    fig, ax = new_figure("onehalf")
    for i in range(6):
        ax.plot([0, 1], [0.2 + i * 0.1, 0.9 - i * 0.05], "o-",
                color=PALETTE[i % 6])
    ax.set_xlim(-0.45, 1.30)
    assert not hit(fig, "比例/概率轴超出")


# --- 回归：run_all 暴露的第三批误伤 ------------------------------------

def test_log_axis_utilization_is_measured_in_log_space():
    """对数轴上的"轴长占比"必须在 log 空间算。

    线性差值除以线性跨度在对数轴上没有意义：facet_metrics 的 log 面板
    数据铺满了整条轴，线性口径却算出 27%。这条 bug 既有的 60% 告警
    档也有，不是新引入的。
    """
    fig, ax = new_figure("onehalf")
    ax.set_xscale("log")
    # facet_metrics 实测形状：失效率跨 4 个数量级 + margins(x=0.22)
    ax.plot([0.002, 0.02, 0.2, 2.0, 20.0], [0, 1, 2, 3, 4], "o",
            color=PALETTE[0])
    ax.margins(x=0.22)
    assert not hit(fig, "被压成一条线")


def test_parallel_coordinates_with_gray_context_lines_is_not_flagged():
    """整束判定要数上灰色背景线：平行坐标常只强调 1–2 条，其余压成灰。"""
    fig, ax = new_figure("onehalf")
    rng = np.random.default_rng(5)
    for _ in range(8):
        ax.plot(range(5), rng.random(5), "-", color="#CCCCCC", lw=0.8)
    for j in range(2):
        ax.plot(range(5), rng.random(5), "-", color=PALETTE[j], lw=1.8)
    assert not hit(fig, "个点用折线连起来")


def test_axis_without_tick_labels_is_not_treated_as_proportion_axis():
    """无刻度的归一化位置轴（平行坐标的 y）谈不上"超出 [0,1]"。

    读者看不到刻度，也就无从误读；轴限留白是给两端数值标注让位。
    """
    fig, ax = new_figure("onehalf")
    rng = np.random.default_rng(7)
    for _ in range(6):
        ax.plot(range(5), rng.random(5), "-", color="#CCCCCC", lw=0.8)
    ax.set_yticks([])
    ax.set_ylim(-0.14, 1.14)
    assert not hit(fig, "比例/概率轴超出")


# --- 回归：比例轴判据的两次收窄 ----------------------------------------

def test_percent_valued_quantity_is_not_a_proportion_axis():
    """φ 体积分数 0.5–1.0 **百分数**恰好落在 [0,1]，但它不是比例。

    实测 q2 的相变密度场 x 轴就是这种量，被误判成概率轴。判据因此
    要求轴标题里确实出现概率/比例/占比/率这类词。
    """
    fig, ax = new_figure("onehalf")
    ax.plot([0.5, 0.6, 0.7, 1.0], [10, 20, 30, 40], "o", color=PALETTE[0])
    ax.set_xlabel("介质A 体积分数 φ / %")
    ax.set_xlim(0.48, 1.23)
    assert not hit(fig, "比例/概率轴超出")


def test_small_margin_on_probability_axis_is_not_flagged():
    """概率轴留 8% 上边距是作者有意为之，不是病。

    matplotlib 默认边距就是 5%；真正的病灶是 q2 原先的 −0.34～1.14
    （下方白留三成）。容差因此定在 ±0.10。
    """
    fig, ax = new_figure("onehalf")
    ax.plot([1, 2, 3, 4], [0.08, 0.22, 0.50, 0.99], "o", color=PALETTE[0])
    ax.set_ylabel("导通概率 P")
    ax.set_ylim(-0.04, 1.08)
    assert not hit(fig, "比例/概率轴超出")


def test_large_slack_on_probability_axis_is_still_flagged():
    fig, ax = new_figure("onehalf")
    ax.plot([1, 2, 3, 4], [0.08, 0.22, 0.50, 0.99], "o", color=PALETTE[0])
    ax.set_ylabel("导通概率 P")
    ax.set_ylim(-0.34, 1.14)
    assert hit(fig, "比例/概率轴超出")


def test_same_label_reported_for_two_conditions_is_not_a_conflict():
    """同一个量名在两个条件下各报一次，不是矛盾。

    实测 q6：图题「KS = 0.0023 通过检验；…KS = 0.1060 拒绝」——两个 KS
    分别属于正确抽样与错误抽样，都由变量生成。只有量名在两侧**各只
    出现一次**、能唯一配对时，数值不一致才算事实矛盾。
    """
    fig, ax = new_figure("onehalf")
    ax.plot([1, 2, 3], [1, 2, 3])
    fig.suptitle("KS = 0.0023 通过检验；错误抽样 KS = 0.1060 被拒绝")
    ax.text(0.1, 0.8, "KS = 0.0023", transform=ax.transAxes,
            bbox=dict(boxstyle="round", fc="w"))
    ax.text(0.1, 0.6, "KS = 0.1060", transform=ax.transAxes,
            bbox=dict(boxstyle="round", fc="w"))
    assert not hit(fig, "两个不同数值")


# ======================================================================
# 独立评审提出的缺陷：先钉测试，再改实现
# ======================================================================

from core import (dot_interval, slope_lines, stat_box,  # noqa: E402
                  callout, ref_line)


def _di_fig(**kw):
    fig, ax = new_figure("onehalf", ratio=0.45)
    fig.subplots_adjust(left=0.2, right=0.6, top=0.85, bottom=0.22)
    ok = dot_interval(ax, ["甲", "乙", "丙", "丁"],
                      [0.08, 0.22, 0.50, 0.99],
                      [0.07, 0.20, 0.47, 0.98],
                      [0.09, 0.24, 0.53, 0.996],
                      threshold=0.90, **kw)
    ax.set_xlabel("导通概率 P")
    return fig, ax, ok


# --- nature 档兼容：两个新构图函数不能在 nature 下必然硬失败 ---

def test_dot_interval_is_usable_in_nature_preset():
    apply_style("nature")
    fig, ax, _ = _di_fig()
    p = probs(fig)
    assert not any("彩色文字" in x for x in p), [x for x in p if "彩色文字" in x]
    assert not any("线宽" in x for x in p), [x for x in p if "线宽" in x]


def test_slope_lines_is_usable_in_nature_preset():
    apply_style("nature")
    fig, ax = new_figure("single")
    slope_lines(ax, ["A", "B", "C"], [3, 5, 4], [4, 4, 6], verdict="2 升 1 降")
    p = probs(fig)
    assert not any("彩色文字" in x for x in p), [x for x in p if "彩色文字" in x]
    assert not any("线宽" in x for x in p), [x for x in p if "线宽" in x]


# --- slope_lines 输入健壮性 ---

def test_slope_lines_rejects_length_mismatch():
    fig, ax = new_figure("single")
    with pytest.raises(ValueError):
        slope_lines(ax, ["A", "B", "C"], [1, 2], [3, 4])


def test_slope_lines_rejects_out_of_range_highlight():
    fig, ax = new_figure("single")
    with pytest.raises(ValueError):
        slope_lines(ax, ["A", "B"], [1, 2], [2, 3], highlight=(99,))


def test_slope_lines_survives_non_finite_input():
    """inf 曾让避让算出 NaN，draw 时抛毫无线索的 StopIteration。"""
    fig, ax = new_figure("single")
    c = slope_lines(ax, ["A", "B", "C"], [1.0, np.inf, 3.0],
                    [2.0, 4.0, np.nan])
    fig.canvas.draw()
    assert c["invalid"] == 2


def test_slope_lines_does_not_count_nan_as_flat():
    """NaN 行被算进 flat，recipe 会把它写进图题，声称"该项无变化"。"""
    fig, ax = new_figure("single")
    c = slope_lines(ax, ["A", "B"], [1.0, np.nan], [2.0, np.nan])
    assert c["flat"] == 0 and c["up"] == 1


# --- dot_interval 边界与返回值语义 ---

def test_dot_interval_with_identical_sizes_does_not_divide_by_zero():
    fig, ax, _ = _di_fig(sizes=[100, 100, 100, 100])
    fig.canvas.draw()
    for ln in ax.lines:
        assert np.isfinite(ln.get_markersize())


def test_dot_interval_single_row_does_not_crash():
    fig, ax = new_figure("onehalf", ratio=0.3)
    ok = dot_interval(ax, ["唯一档"], [0.5], [0.4], [0.6], threshold=0.45)
    fig.canvas.draw()
    assert len(ok) == 1 and bool(ok[0]) is False


def test_dot_interval_returns_order_so_callers_can_map_back():
    """sort=True 重排了行，只返回排序后的 ok 会让调用方索引错档位。"""
    fig, ax = new_figure("onehalf", ratio=0.45)
    ok, order = dot_interval(ax, ["高", "低", "中"], [0.9, 0.1, 0.5],
                             [0.85, 0.05, 0.45], [0.95, 0.15, 0.55],
                             threshold=0.8, return_order=True)
    labs = [t.get_text() for t in ax.get_yticklabels()]
    assert labs == ["低", "中", "高"]
    assert list(order) == [1, 2, 0]
    assert bool(ok[-1]) is True and bool(ok[0]) is False


# --- 两条"自欺"的检查 ---

def test_unexplained_band_still_fires_when_a_stat_box_is_present():
    """has_key 把任意 ax.texts 当"已解释"，而 QA 强制每图必须有 stat_box
    （走 ax.text）——于是凡能过 QA 的图，这条检查必然被自己静音。"""
    fig, ax = new_figure("onehalf")
    x = np.linspace(0, 1, 20)
    ax.plot(x, x)
    ax.fill_between(x, x - 0.1, x + 0.1, alpha=0.2)
    stat_box(ax, ["n = 20", "RMS = 0.10"], loc="upper left")
    assert hit(fig, "无法得知其含义")


def test_evenly_spaced_continuous_scan_may_be_joined_by_a_line():
    """等距连续量的参数扫描连线完全合法——数模里最常见的形状之一。

    离散档位（φ = 0.5/0.6/0.7/1.00，不等距）才是 sparse_line 要抓的。
    """
    fig, ax = new_figure("onehalf")
    ax.plot([0.0, 0.1, 0.2, 0.3, 0.4], [12.0, 9.5, 7.1, 5.4, 4.2], "o-",
            color=PALETTE[0])
    ax.set_xlabel("间隙 D0（m）")
    assert not hit(fig, "个点用折线连起来")


def test_unequally_spaced_report_levels_are_still_flagged():
    fig, ax = new_figure("onehalf")
    ax.plot([0.5, 0.6, 0.7, 1.0], [0.081, 0.216, 0.497, 0.9935], "o-",
            color=PALETTE[0])
    assert hit(fig, "个点用折线连起来")


# --- 直标互压：cohort 图上肉眼可见的叠字，QA 全绿放行 ---

def test_two_bare_labels_overlapping_each_other_is_flagged():
    fig, ax = new_figure("onehalf")
    ax.plot([0, 1], [0, 1])
    ax.annotate("站点31 57.7", xy=(0.5, 0.5), fontsize=9)
    ax.annotate("站点09 56.4", xy=(0.5, 0.502), fontsize=9)
    assert hit(fig, "直标互相重叠")


def test_unknown_allow_code_is_rejected():
    """allow 不校验时，拼错的码静默无效：用户以为豁免了，图照样被拦。"""
    fig, ax = new_figure("onehalf")
    ax.plot([1, 2, 3], [1, 2, 3])
    with pytest.raises(ValueError):
        run_qa(fig, strict=False, allow=("overlaps",))


def test_proportion_axis_locked_to_zero_one_is_not_called_outlier_stretched():
    """两条检查互相打架：check 13 要求锁 [0,1]，锁完 check 8b 又说
    "数据被压成一条线（多半是单个离群点把轴撑爆）"——诊断词与实情相反。"""
    fig, ax = new_figure("onehalf")
    ax.plot([1, 2, 3, 4], [0.42, 0.47, 0.53, 0.58], "o", color=PALETTE[0])
    ax.set_ylabel("命中率")
    ax.set_ylim(0, 1)
    assert not hit(fig, "被压成一条线")


def test_no_check_is_silently_skipped_on_a_normal_figure(capsys):
    """qa.py 几乎每块检查都包在 except 里，抛异常就静默跳过。

    后果是所有"不该触发"的测试会因为"检查根本没跑"而假绿。这条守住：
    正常图上不允许出现任何 "[QA note] …未执行"。
    """
    fig, ax = new_figure("onehalf")
    x = np.linspace(0, 1, 40)
    ax.plot(x, np.sin(x * 6), color=PALETTE[0], label="曲线")
    ax.fill_between(x, np.sin(x * 6) - 0.1, np.sin(x * 6) + 0.1,
                    alpha=0.2, label="95% CI")
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.suptitle("结论句：某量在 x=0.5 处达峰")
    stat_box(ax, ["n = 40"], loc="upper left")
    run_qa(fig, strict=False)
    out = capsys.readouterr().out
    skipped = [ln for ln in out.splitlines() if "未执行" in ln]
    assert not skipped, skipped


def test_strict_mode_blocks_save_on_a_bad_figure():
    """"坏图不落盘"是整个 skill 的核心保障，此前零覆盖。"""
    from core.style import is_styled       # noqa: F401
    fig, ax = new_figure("onehalf")
    ax.plot([0.5, 0.6, 0.7, 1.0], [0.081, 0.216, 0.497, 0.9935], "o-",
            color=PALETTE[0])
    with pytest.raises(AssertionError):
        run_qa(fig, strict=True)
    assert not getattr(fig, "_ff_qa_ok", False)


# --- 收尾：评审 P2 ---------------------------------------------------

def test_value_column_does_not_overflow_the_canvas_by_default():
    """`value_col=True` 的数值列锚在轴外 1.06 且 clip 关闭。

    不预留右侧版面就会画出画布（实测溢出 9%），而这个义务此前只写在
    docstring 里——函数应当自己让位。
    """
    fig, ax = new_figure("onehalf", ratio=0.45)
    dot_interval(ax, ["甲", "乙", "丙"], [0.2, 0.5, 0.8],
                 [0.15, 0.45, 0.75], [0.25, 0.55, 0.85], threshold=0.6)
    fig.canvas.draw()
    rd = fig.canvas.get_renderer()
    right = max(t.get_window_extent(rd).x1 for t in ax.texts)
    assert right <= fig.bbox.x1 + 1, (right, fig.bbox.x1)


def test_cohort_context_lines_are_printable():
    """cohort 的群体线原为 0.72@0.45，与白底对比度 1.33:1，300dpi 印刷后
    几乎全白——它们仍然承载数据，不能淡到看不见。"""
    fig, ax = new_figure("single")
    rng = np.random.default_rng(1)
    b = rng.normal(60, 8, 20)
    slope_lines(ax, [f"S{i}" for i in range(20)], b, b + rng.normal(1, 5, 20),
                highlight=(0,), mode="cohort")
    from core.colors import luminance
    ctx = [ln for ln in ax.lines if not _is_colored_hex(ln.get_color())]
    assert ctx, "没有中性色群体线"
    eff = []
    for ln in ctx:
        g = float(matplotlib.colors.to_rgb(ln.get_color())[0])
        a = ln.get_alpha() if ln.get_alpha() is not None else 1.0
        eff.append(1.0 - a * (1.0 - g))       # 叠在白底上的等效灰度
    worst = max(eff)
    ratio = 1.05 / (luminance((worst, worst, worst)) + 0.05)
    assert ratio >= 2.8, f"对比度仅 {ratio:.2f}:1（等效灰度 {worst:.3f}）"


def _is_colored_hex(c):
    r, g, b = matplotlib.colors.to_rgb(c)
    return max(r, g, b) - min(r, g, b) > 0.06


def test_incommensurable_series_sharing_one_axis_is_flagged():
    """SPEC §2.2 把"不可通约的量共用 y 轴"列为硬伤，但 qa 只查了 twinx，
    同一根轴上直接叠两个量级差 100× 的量一直漏网。"""
    fig, ax = new_figure("onehalf")
    x = np.linspace(0, 1, 30)
    ax.plot(x, 0.2 + 0.6 * x, color=PALETTE[0])           # 比例 0–1
    ax.plot(x, 1200 + 400 * x, color=PALETTE[1])          # 元，量级差 1000×
    assert hit(fig, "不可通约")


def test_series_on_comparable_scales_are_not_flagged():
    fig, ax = new_figure("onehalf")
    x = np.linspace(0, 1, 30)
    ax.plot(x, 20 + 30 * x, color=PALETTE[0])
    ax.plot(x, 45 + 25 * x, color=PALETTE[1])
    assert not hit(fig, "不可通约")


def test_log_axis_spanning_decades_is_not_flagged_as_incommensurable():
    """对数轴上跨几个数量级是正当的，那正是用 log 轴的理由。"""
    fig, ax = new_figure("onehalf")
    x = np.linspace(0, 1, 30)
    ax.set_yscale("log")
    ax.plot(x, 0.01 * (1 + x), color=PALETTE[0])
    ax.plot(x, 100 * (1 + x), color=PALETTE[1])
    assert not hit(fig, "不可通约")


# --- 第 2 轮评审：sparse_line 判据整个搞错了 ---------------------------

def test_categorical_positions_joined_by_a_line_is_flagged():
    """离散方案几乎总画在整数 x 上，而整数**天然等距**——等距豁免正好
    把这条检查要抓的最典型病灶放行了。判据必须看刻度是不是类别名。"""
    fig, ax = new_figure("onehalf")
    ax.plot([0, 1, 2, 3], [72.1, 65.8, 88.4, 59.2], "o-", color=PALETTE[0])
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["方案A", "方案B", "方案C", "方案D"])
    assert hit(fig, "个点用折线连起来")


def test_geometric_scan_is_not_flagged():
    """网格加密倍数 / 样本量倍增（1,2,4,8,16）在数模里极常见，
    它不等距，但等比——同样是连续量扫描，连线合法。"""
    fig, ax = new_figure("onehalf")
    ax.plot([1, 2, 4, 8, 16], [10, 5.2, 2.7, 1.4, 0.8], "o-",
            color=PALETTE[0])
    ax.set_xlabel("网格加密倍数")
    assert not hit(fig, "个点用折线连起来")


# --- dot_interval：面积声明与 marker 吞区间 ---------------------------

def test_marker_area_is_affine_in_sizes():
    """图内写"点面积 ∝ N_A"而 ms 是**直径**线性：实测面积比 3.16×，
    N_A 之比 2.00×，读者按声明反解会高估 58%——事实错误级。"""
    fig, ax = new_figure("onehalf", ratio=0.5)
    sz = np.array([354.0, 425.0, 496.0, 707.0])
    dot_interval(ax, list("甲乙丙丁"), [0.2, 0.4, 0.6, 0.8],
                 [0.15, 0.35, 0.55, 0.75], [0.25, 0.45, 0.65, 0.85],
                 sizes=sz, sort=False, value_col=False)
    ms = np.array([ln.get_markersize() for ln in ax.lines
                   if ln.get_marker() == "o"])
    assert ms.size == 4
    a = ms ** 2
    norm_a = (a - a.min()) / np.ptp(a)
    norm_s = (sz - sz.min()) / np.ptp(sz)
    assert np.allclose(norm_a, norm_s, atol=0.02), (norm_a, norm_s)


def test_marker_does_not_swallow_a_narrow_interval():
    """结论那一行区间最窄（1.66pt）却被 8pt 实心点完全盖住——整张图要
    论证的"下界过线"在图上看不见。窄于 marker 时应改空心并让区间压在上层。"""
    fig, ax = new_figure("onehalf", ratio=0.5)
    dot_interval(ax, ["宽", "窄"], [0.30, 0.90], [0.20, 0.8985],
                 [0.40, 0.9015], threshold=0.85, sort=False,
                 value_col=False)
    fig.canvas.draw()
    pts = [ln for ln in ax.lines if ln.get_marker() == "o"]
    assert len(pts) == 2
    narrow = pts[1]
    assert narrow.get_markerfacecolor() in ("white", "w"), \
        narrow.get_markerfacecolor()


def test_dot_interval_rejects_mismatched_sizes_length():
    fig, ax = new_figure("onehalf", ratio=0.4)
    with pytest.raises(ValueError):
        dot_interval(ax, list("甲乙丙"), [0.2, 0.4, 0.6],
                     [0.1, 0.3, 0.5], [0.3, 0.5, 0.7], sizes=[1, 2])


def test_dot_interval_reports_non_finite_rows():
    """NaN 行此前静默变成"未达标"，还在数值列里印出字面量 nan。"""
    fig, ax = new_figure("onehalf", ratio=0.45)
    ok, order = dot_interval(ax, list("甲乙丙"), [0.5, np.nan, 0.8],
                             [0.4, np.nan, 0.75], [0.6, np.nan, 0.85],
                             threshold=0.7, return_order=True)
    fig.canvas.draw()
    assert not any("nan" in t.get_text().lower() for t in ax.texts)


# --- 豁免留痕契约必须对所有码成立 ---

def test_waiving_grouped_bars_leaves_a_trace():
    """docstring 承诺"豁免会记入 fig._ff_qa_waived 并标进文件名"，
    而它举例用的 grouped_bars 恰恰是纯跳过、连 [QA WAIVED] 都不打。"""
    fig, ax = new_figure("onehalf")
    ax.bar([0, 1, 2], [3, 4, 5], width=0.35)
    ax.bar([0.35, 1.35, 2.35], [2, 3, 4], width=0.35)
    run_qa(fig, strict=False, allow=("grouped_bars",))
    assert getattr(fig, "_ff_qa_waived", None), "豁免没有留痕"


# --- 直标文字对白底的对比度 ---

def test_slope_end_labels_are_readable_on_white():
    """semantic("bad")=#e69f00 对白底仅 2.25:1；斜率图六个下降方向的
    端点直标全是这个色，比上一轮修掉的 1.33:1 好不了多少。"""
    fig, ax = new_figure("single")
    slope_lines(ax, ["A", "B", "C"], [72, 66, 61], [69, 64, 55])
    from core.colors import luminance
    worst = 21.0
    for t in ax.texts:
        if not t.get_text().strip():
            continue
        L = luminance(t.get_color())
        worst = min(worst, 1.05 / (L + 0.05))
    assert worst >= 3.0, f"最差直标对比度 {worst:.2f}:1"


def test_value_column_does_not_intrude_into_a_neighbour_panel():
    """自动让位只比对画布右缘，不知道右边有邻居——多面板下数值列会直接
    骑进隔壁面板，而 5c"压数据"只遍历带框注释，无框直标兜不住。"""
    import matplotlib.pyplot as _plt
    fig, axes = _plt.subplots(1, 2, figsize=(5.35, 2.2))
    dot_interval(axes[0], ["甲", "乙", "丙"], [0.2, 0.5, 0.8],
                 [0.15, 0.45, 0.75], [0.25, 0.55, 0.85], threshold=0.6)
    fig.canvas.draw()
    rd = fig.canvas.get_renderer()
    right = max(t.get_window_extent(rd).x1 for t in axes[0].texts)
    nb_x0 = axes[1].get_window_extent(rd).x0
    assert right <= nb_x0 + 1, (right, nb_x0)


# --- 第 3 轮评审：三处"测试绕开了失效区" -------------------------------

def test_geometric_scan_on_a_log_axis_is_not_flagged():
    r"""几何扫描的**标准画法**就是 log 轴，而 log 刻度是 mathtext
    `$\mathdefault{10^{0}}$`，`float()` 解析不了 → 被判成类别轴 →
    等比豁免根本走不到 → 硬拒绝。上一条测试用线性轴，正好绕开。"""
    fig, ax = new_figure("onehalf")
    ax.plot([1, 2, 4, 8, 16], [10, 5.2, 2.7, 1.4, 0.8], "o-",
            color=PALETTE[0])
    ax.set_xscale("log")
    ax.set_xlabel("网格加密倍数")
    assert not hit(fig, "个点用折线连起来")


def test_evenly_spaced_dates_are_not_flagged():
    """等间隔的日期轴同样是连续量，刻度文本却解析不成 float。"""
    import datetime as _dt
    fig, ax = new_figure("onehalf")
    xs = [_dt.date(2026, 1, 5) + _dt.timedelta(days=7 * i) for i in range(5)]
    ax.plot(xs, [12, 9.5, 7.1, 5.4, 4.2], "o-", color=PALETTE[0])
    assert not hit(fig, "个点用折线连起来")


def test_integer_positions_without_tick_labels_are_still_flagged():
    """作者忘了 set_xticklabels 时，四个方案画在 0,1,2,3 上仍是离散档位。
    连续量不会恰好只在小整数上取 3–6 个样本。"""
    fig, ax = new_figure("onehalf")
    ax.plot([0, 1, 2, 3], [72.1, 65.8, 88.4, 59.2], "o-", color=PALETTE[0])
    assert hit(fig, "个点用折线连起来")


def test_bare_label_covering_a_panel_title_is_flagged():
    """5b3 把标题整类从 _bare 剔除后，"无框直标压面板标题"从有覆盖变成
    零覆盖——而面板标题按本项目的规矩就是结论句。"""
    fig, ax = new_figure("onehalf")
    ax.plot([1, 2, 3], [1, 2, 3])
    ax.set_title("方案 C 成本最低")
    # 标题在坐标区**之上**，用 axes 坐标才放得上去（数据坐标会被 clip
    # 成退化 bbox，测试就测了个寂寞）
    ax.annotate("压住标题的直标", xy=(0.35, 1.02), xycoords="axes fraction",
                fontsize=9)
    assert hit(fig, "压住") or hit(fig, "重叠")


def test_narrow_interval_detection_uses_settled_limits():
    """_tight 判定时 viewLim 还停在默认 (0,1)，与真实渲染宽度无关：
    量级 1200 的极窄区间（0.45pt）判成"不窄"仍用实心点吞掉证据。"""
    fig, ax = new_figure("onehalf", ratio=0.4)
    dot_interval(ax, list("甲乙丙"), [1200., 1500., 1800.],
                 [1199.5, 1499.5, 1799.5], [1200.5, 1500.5, 1800.5],
                 value_col=False, sort=False)
    fig.canvas.draw()
    pts = [ln for ln in ax.lines if ln.get_marker() == "o"]
    assert all(p.get_markerfacecolor() in ("white", "w") for p in pts), \
        [p.get_markerfacecolor() for p in pts]


def test_wide_interval_keeps_a_solid_marker():
    """反向：量程 0.007 时区间渲染 38.8pt（几乎横跨面板），却被判成
    "比标记还窄"而全部转空心。"""
    fig, ax = new_figure("onehalf", ratio=0.4)
    dot_interval(ax, list("甲乙丙"), [0.001, 0.004, 0.007],
                 [0.0005, 0.0035, 0.0065], [0.0015, 0.0045, 0.0075],
                 threshold=0.006, value_col=False, sort=False)
    fig.canvas.draw()
    pts = [ln for ln in ax.lines if ln.get_marker() == "o"]
    ok_pt = pts[-1]
    assert ok_pt.get_markerfacecolor() not in ("white", "w")


def test_end_labels_are_readable_on_white():
    """_ink 只接了 3/6 个着色点：end_labels/callout/end_label 仍吐原色，
    实测 #E69F00 = 2.25:1。8 张 gallery 图受影响。"""
    from core import end_labels
    from core.colors import luminance
    fig, ax = new_figure("onehalf")
    ax.plot([0, 1], [1, 2])
    end_labels(ax, [(1, 2, "方案A", "#E69F00"), (1, 1.5, "方案B", "#56B4E9")])
    worst = 21.0
    for t in ax.texts:
        if t.get_text().strip():
            worst = min(worst, 1.05 / (luminance(t.get_color()) + 0.05))
    assert worst >= 3.0, f"最差 {worst:.2f}:1"


def test_sizes_with_nan_does_not_silently_hide_every_marker():
    """sz.min()/ptp 被 NaN 传染 → 每行 ms=nan → 所有点不渲染，无警告。"""
    fig, ax = new_figure("onehalf", ratio=0.45)
    with pytest.raises(ValueError):
        dot_interval(ax, list("甲乙丙丁"), [0.2, 0.4, 0.6, 0.8],
                     [0.1, 0.3, 0.5, 0.7], [0.3, 0.5, 0.7, 0.9],
                     sizes=[1, 2, np.nan, 4])


def test_inside_value_column_uses_uniform_precision():
    """N13 只修了 value_col=True 一半，"inside" 分支仍是 :.3g。"""
    fig, ax = new_figure("onehalf", ratio=0.45)
    dot_interval(ax, list("甲乙"), [0.994, 0.0698], [0.98, 0.06],
                 [0.999, 0.08], value_col="inside", sort=False)
    txt = [t.get_text() for t in ax.texts if t.get_text().strip()]
    dec = {len(t.split(".")[1]) for t in txt if "." in t}
    assert len(dec) == 1, txt


# --- 第 4 轮评审 -------------------------------------------------------

def test_small_integer_continuous_quantity_is_not_flagged():
    """k-means 肘部 k=2..6、迭代次数 1..5、多项式阶数 1..4 都是小整数上的
    **连续量**，插值合法。`max<12 的整数即位置编码`判过头，把数模里最常见
    的一类曲线硬挡了（sparse_line 是硬错，直接拦 save_figure）。"""
    for xs, lab in ([[2, 3, 4, 5, 6], "簇数 k"],
                    [[1, 2, 3, 4, 5], "迭代次数"],
                    [[1, 2, 3, 4], "多项式阶数"]):
        fig, ax = new_figure("onehalf")
        ax.plot(xs, np.linspace(120, 35, len(xs)), "-o", color=PALETTE[0])
        ax.set_xlabel(lab)
        assert not hit(fig, "个点用折线连起来"), lab
        plt.close(fig)


def test_value_column_respects_neighbour_tight_bbox():
    """判据用坐标区盒、度量用 tightbbox，两个口径不一致：邻居的 ylabel
    往左伸出后 tightbbox.x0 跑到本轴右缘左边，邻居被整个排除，让位一次都
    不收缩——比不改还差。断言必须用 tightbbox，否则测不出这个回归。"""
    import matplotlib.pyplot as _plt
    fig, axes = _plt.subplots(1, 2, figsize=(5.35, 2.2))
    axes[1].set_ylabel("总成本 / 元")
    axes[1].plot([1, 2], [1, 2])
    dot_interval(axes[0], ["甲", "乙", "丙"], [0.2, 0.5, 0.8],
                 [0.15, 0.45, 0.75], [0.25, 0.55, 0.85], threshold=0.6)
    fig.canvas.draw()
    rd = fig.canvas.get_renderer()
    right = max(t.get_window_extent(rd).x1 for t in axes[0].texts)
    assert right <= axes[1].get_tightbbox(rd).x0 + 1, \
        f"侵入邻居 {right - axes[1].get_tightbbox(rd).x0:+.1f}px"


def test_value_column_survives_a_hidden_panel():
    """`Axes.get_tightbbox()` 对不可见轴返回 None（不抛异常），
    `_ob.x0` 的 AttributeError 被外层 except 吞成一行 note，整个收缩
    循环被跳过——而"网格里隐藏没用到的格子"正是小倍数的标准写法。"""
    import matplotlib.pyplot as _plt
    fig, ax = _plt.subplots(2, 2, figsize=(5.35, 3))
    ax[1, 1].set_visible(False)
    dot_interval(ax[0, 1], ["甲", "乙"], [0.3, 0.7], [0.2, 0.6],
                 [0.4, 0.8], threshold=0.5)
    fig.canvas.draw()
    right = max(t.get_window_extent(fig.canvas.get_renderer()).x1
                for t in ax[0, 1].texts)
    assert right <= fig.bbox.x1 + 1, f"溢出画布 {right - fig.bbox.x1:+.1f}px"


def test_scalar_sizes_raises_value_error():
    """报错文案自己先炸：`len(np.asarray(5))` 对 0-d 数组非法。"""
    fig, ax = new_figure("onehalf", ratio=0.4)
    with pytest.raises(ValueError):
        dot_interval(ax, list("甲乙丙"), [0.2, 0.4, 0.6],
                     [0.1, 0.3, 0.5], [0.3, 0.5, 0.7], sizes=5)


def test_nature_preset_can_produce_a_passing_figure():
    """nature 档 figure.titlesize 留在 matplotlib 默认 large=8.4pt，而 qa
    硬拒 >7pt；不加图题又报"无图题"——该档实际不可用，只因 gallery 全是
    cn 档才从没暴露。"""
    apply_style("nature")
    fig, ax = new_figure("single")
    x = np.linspace(0, 1, 40)
    ax.plot(x, np.sin(x * 6), color=PALETTE[0])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.suptitle("正弦在 x=0.26 处达峰")
    stat_box(ax, ["n = 40"], loc="upper right")
    assert not hit(fig, "7pt")


def test_value_column_falls_back_when_magnitudes_span_decades():
    """按最大值定位数会把小量级整行印成 0（比位数参差更伤：静默印错）。"""
    fig, ax = new_figure("onehalf", ratio=0.4)
    dot_interval(ax, ["大", "小"], [1.2e6, 3.4e-3], [1.1e6, 3.0e-3],
                 [1.3e6, 3.8e-3], sort=False)
    txt = [t.get_text() for t in ax.texts if "[" in t.get_text()]
    assert not any(t.startswith("0 [0, 0]") for t in txt), txt


def test_qa_flags_low_contrast_direct_labels():
    """recipe 里裸 `ax.annotate(color=PALETTE[...])` 绕开 core.annotate，
    7 张 gallery 图仍有 2.25:1 的直标。qa 里一条文字对比度检查都没有，
    这类缺陷结构上不可能被自动发现。"""
    fig, ax = new_figure("onehalf")
    ax.plot([1, 2, 3], [1, 2, 3])
    ax.annotate("发虚的直标", xy=(2, 2), color="#E69F00", fontsize=8)
    assert hit(fig, "对比度")


def test_qa_does_not_flag_readable_labels():
    fig, ax = new_figure("onehalf")
    ax.plot([1, 2, 3], [1, 2, 3])
    ax.annotate("清晰的直标", xy=(2, 2), color="#0072B2", fontsize=8)
    assert not hit(fig, "对比度")


def test_white_text_on_a_dark_cell_is_not_flagged():
    """深底白字是正确做法（热力图注数、条内直标），判据不能假定白底。"""
    fig, ax = new_figure("onehalf")
    # viridis 高值端是浅黄绿，白字在那上面确实不可读——只在深色端放白字，
    # 这正是 annotated_heatmap.py 的做法（按格子亮度选黑/白）
    ax.imshow(np.full((4, 4), 0.05), cmap="viridis", vmin=0, vmax=1)
    for i in range(4):
        ax.text(i, i, "0.05", color="white", ha="center", va="center")
    assert not hit(fig, "对比度")


def test_white_text_inside_a_dark_bar_is_not_flagged():
    fig, ax = new_figure("onehalf")
    ax.barh([0, 1, 2], [3, 5, 4], color="#0072B2")
    for i, v in enumerate([3, 5, 4]):
        ax.text(v * 0.5, i, f"{v}", color="white", ha="center", va="center")
    assert not hit(fig, "对比度")


def test_custom_tick_labels_on_numeric_positions_are_flagged():
    """`set_xticklabels` 在 mpl 3.10 装的是 FuncFormatter 不是
    FixedFormatter，所以那半个判据是死代码：方案画在 [10,20,30,40] 上
    配自定义标签会漏检。"""
    fig, ax = new_figure("onehalf")
    ax.plot([10, 20, 30, 40], [72.1, 65.8, 88.4, 59.2], "o-",
            color=PALETTE[0])
    ax.set_xticks([10, 20, 30, 40])
    ax.set_xticklabels(["方案A", "方案B", "方案C", "方案D"])
    assert hit(fig, "个点用折线连起来")


# --- 第 5 轮评审：text_contrast 的豁免用几何冒充颜色 -------------------

def test_white_text_on_a_contourf_field_is_not_flagged():
    """`ContourSet.get_window_extent()` 返回 Bbox(inf,inf,-inf,-inf)，
    几何豁免整个失效 → 场图白字被硬拒。而 contour_field 就是本库的
    场图 recipe。"""
    fig, ax = new_figure("onehalf")
    X, Y = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20))
    ax.contourf(X, Y, X * Y, levels=8, cmap="viridis")
    ax.text(0.2, 0.2, "0.04", color="white", ha="center")
    assert not hit(fig, "对比度")


def test_white_text_on_a_pale_segment_is_flagged():
    """反向：白字落进**浅色**段被无条件豁免。gallery 的
    composition_stacked 里就躺着 7 处这样的真缺陷（实测 1.64:1），
    而检查看不见——它只判"有没有被图元覆盖"，不判底色深浅。"""
    fig, ax = new_figure("onehalf")
    ax.barh([0], [0.6], color="#9ecae1")
    ax.text(0.3, 0, "62%", color="white", ha="center", va="center")
    assert hit(fig, "对比度")


def test_text_over_an_opaque_white_bbox_is_still_checked():
    """只看 bbox 的 alpha 不看颜色：白底框里的浅橙字对白框仍不合格。"""
    fig, ax = new_figure("onehalf")
    ax.plot([1, 2, 3], [1, 2, 3])
    ax.annotate("白框里的浅橙字", xy=(2, 2), color="#E69F00", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    assert hit(fig, "对比度")


def test_figure_level_legend_text_is_inspected():
    """`_all_texts` 漏掉 fig.legends → 字号下限 / 豆腐块 / nature 彩色文字
    / 对比度**全部**对它失明。"""
    from core.qa import _all_texts
    fig, ax = new_figure("onehalf")
    ax.plot([1, 2], [1, 2], label="曲线")
    fig.legend(fontsize=3)
    assert any("曲线" in t.get_text() for t in _all_texts(fig))


def test_three_d_axis_labels_are_inspected():
    """3D 的 zlabel 与 z 刻度整条不在视野内，而 surface3d_project 是在产
    recipe 且有 zlabel。"""
    from core.qa import _all_texts
    import matplotlib.pyplot as _plt
    fig = _plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_zlabel("高程 z")
    assert any("高程" in t.get_text() for t in _all_texts(fig))


def test_unit_formatter_on_a_linear_axis_is_not_a_category_axis():
    """`_non_numeric` 把 EngFormatter/千分位/带单位后缀全判成类别轴 →
    等距连续扫描被硬拒。注释自己写着"不要去解析渲染后的刻度文本"。"""
    from matplotlib.ticker import EngFormatter, FuncFormatter
    for name, fmt in [("Eng", EngFormatter(unit="Hz")),
                      ("单位后缀", FuncFormatter(lambda v, p: f"{v:g} km")),
                      ("货币", FuncFormatter(lambda v, p: f"${v:.0f}"))]:
        fig, ax = new_figure("onehalf")
        ax.plot([5, 10, 15, 20, 25], [12, 9.5, 7.1, 5.4, 4.2], "o-",
                color=PALETTE[0])
        ax.xaxis.set_major_formatter(fmt)
        assert not hit(fig, "个点用折线连起来"), name
        plt.close(fig)


def test_value_column_warns_when_it_cannot_fit():
    """让位撞到 0.15 下限后静默放弃：用户拿到数值列骑进邻居的图，全程
    无告警，而 QA 的遮挡检查只遍历带框注释，兜不住无框直标。"""
    import matplotlib.pyplot as _plt
    import io as _io
    import contextlib
    fig, axes = _plt.subplots(1, 3, figsize=(5.35, 1.8))
    for a in axes[1:]:
        a.set_ylabel("总成本 / 元")
        a.plot([1, 2], [1, 2])
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        dot_interval(axes[0], ["甲", "乙", "丙"], [0.2, 0.5, 0.8],
                     [0.15, 0.45, 0.75], [0.25, 0.55, 0.85], threshold=0.6)
        fig.canvas.draw()
    assert "越界" in buf.getvalue() or "让位" in buf.getvalue(), buf.getvalue()


def test_text_with_a_white_halo_on_a_field_is_not_flagged():
    """白色描边光晕就是"文字压在场图上"的正解（annotate.py 的 `_halo`
    专为此写）。采样只看 bbox 中位背景会把它算成低对比度。"""
    import matplotlib.patheffects as _pe
    fig, ax = new_figure("onehalf")
    X, Y = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20))
    ax.contourf(X, Y, X * Y, levels=8, cmap="YlOrRd")
    t = ax.text(0.5, 0.5, "峰值 0.87", color="#3D7A6B", fontsize=8)
    t.set_path_effects([_pe.withStroke(linewidth=2.4, foreground="white")])
    assert not hit(fig, "对比度")


def test_text_without_a_halo_on_a_field_is_still_flagged():
    fig, ax = new_figure("onehalf")
    X, Y = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20))
    ax.contourf(X, Y, X * Y, levels=8, cmap="YlOrRd")
    ax.text(0.5, 0.5, "峰值 0.87", color="#E69F00", fontsize=8)
    assert hit(fig, "对比度")


# --- 第 6 轮评审：像素采样的两个旁路 -----------------------------------

def test_constrained_layout_does_not_drift_during_probe():
    """藏字重绘会让 layout engine 重新求解，坐标区扩张（实测 x0 从 0.077
    漂到 0.021），于是背景采自**另一套版面**，而文字 extent 是恢复后量的
    ——黑刻度标签在白底上被判 1.65:1，一张完全可读的图被硬拒。"""
    fig, ax = new_figure("onehalf", ratio=0.5, layout="constrained")
    ax.barh(["甲", "乙", "丙"], [0.62, 0.44, 0.31], color="#08306b")
    ax.set_ylabel("很长很长很长的分组名称轴")
    assert not hit(fig, "对比度")


def test_tight_layout_does_not_drift_during_probe():
    fig, ax = new_figure("onehalf", ratio=0.5, layout="tight")
    ax.barh(["甲", "乙", "丙"], [0.62, 0.44, 0.31], color="#08306b")
    ax.set_ylabel("很长很长很长的分组名称轴")
    assert not hit(fig, "对比度")


def test_a_no_op_path_effect_does_not_grant_exemption():
    """`pe.Normal()` 不画任何描边，却因为没有 `_gc` 属性走 fail-open 分支
    被豁免——一行 shipped matplotlib 惯用法即可静默绕过一条硬拒检查。"""
    import matplotlib.patheffects as _pe
    fig, ax = new_figure("onehalf")
    ax.plot([1, 2, 3], [1, 2, 3])
    t = ax.annotate("发虚的直标", xy=(2, 2), color="#E69F00", fontsize=8)
    t.set_path_effects([_pe.Normal()])
    assert hit(fig, "对比度")


def test_stroke_without_foreground_does_not_grant_exemption():
    """不给 foreground 时 mpl 用**文字自身色**描边＝把过浅的字加粗，
    不是光晕。"""
    import matplotlib.patheffects as _pe
    fig, ax = new_figure("onehalf")
    ax.plot([1, 2, 3], [1, 2, 3])
    t = ax.annotate("发虚的直标", xy=(2, 2), color="#E69F00", fontsize=8)
    t.set_path_effects([_pe.withStroke(linewidth=2.5)])
    assert hit(fig, "对比度")


def test_white_text_with_white_halo_is_still_flagged():
    """白字 + 白光晕压白底完全不可见，而旧判据不看文字自身颜色。"""
    import matplotlib.patheffects as _pe
    fig, ax = new_figure("onehalf")
    ax.plot([1, 2, 3], [1, 2, 3])
    t = ax.annotate("看不见的字", xy=(2, 2), color="white", fontsize=8)
    t.set_path_effects([_pe.withStroke(linewidth=2.4, foreground="white")])
    assert hit(fig, "对比度")


def test_offset_text_is_inspected():
    """`ticklabel_format(style="sci")` 一触发就出现的 offset text 不在
    `_all_texts` 视野内——字号下限/豆腐块/nature 禁彩色/对比度全部失明。"""
    from core.qa import _all_texts
    fig, ax = new_figure("onehalf")
    ax.plot([1e6, 2e6], [1, 2])
    ax.ticklabel_format(style="sci", scilimits=(0, 0))
    fig.canvas.draw()
    ot = ax.xaxis.get_offset_text()
    assert ot.get_text().strip()
    assert any(t is ot for t in _all_texts(fig))


def test_legend_title_is_inspected():
    from core.qa import _all_texts
    fig, ax = new_figure("onehalf")
    ax.plot([1, 2], [1, 2], label="曲线")
    ax.legend(title="方案组")
    assert any("方案组" in t.get_text() for t in _all_texts(fig))


# --- 第 7 轮：描边豁免在"退化取值"上仍 fail-open ----------------------

def _dark_field_fig(color, effects=None, text="压在黑场上"):
    fig, ax = new_figure("onehalf")
    ax.imshow(np.zeros((6, 6)), cmap="gray", vmin=0, vmax=1)
    t = ax.text(2.5, 2.5, text, color=color, fontsize=8, ha="center")
    if effects:
        t.set_path_effects(effects)
    return fig


def test_hairline_halo_does_not_grant_exemption():
    """`linewidth=0.1` 画不出一圈可见光晕，但旧判据只看 foreground 键
    存不存在，linewidth 完全没参与——一个关键字参数就能关掉硬拒检查。"""
    import matplotlib.patheffects as _pe
    fig = _dark_field_fig("black", [_pe.withStroke(linewidth=0.1,
                                                   foreground="white")])
    assert hit(fig, "对比度")


def test_transparent_halo_does_not_grant_exemption():
    """`foreground=(1,1,1,0)` 全透明，`to_rgb` 又把 alpha 丢了。"""
    import matplotlib.patheffects as _pe
    fig = _dark_field_fig("black", [_pe.withStroke(linewidth=3,
                                                   foreground=(1, 1, 1, 0.0))])
    assert hit(fig, "对比度")


def test_none_halo_does_not_grant_exemption():
    """`to_rgb("none")` 返回 (0,0,0)——白字压白底会被当成"有黑光晕"放行。"""
    import matplotlib.patheffects as _pe
    fig, ax = new_figure("onehalf")
    ax.plot([1, 2, 3], [1, 2, 3])
    t = ax.annotate("看不见的白字", xy=(2, 2), color="white", fontsize=8)
    t.set_path_effects([_pe.withStroke(linewidth=3, foreground="none")])
    assert hit(fig, "对比度")


def test_proper_halo_still_grants_exemption():
    """真正的白光晕深字仍要放行，否则 contour_field 会被误杀。"""
    import matplotlib.patheffects as _pe
    fig = _dark_field_fig("black", [_pe.withStroke(linewidth=2.4,
                                                   foreground="white")])
    assert not hit(fig, "对比度")


def test_figure_level_legend_title_is_inspected():
    """axes 图例补了 get_title()，同一函数上面五行的 fig.legends 循环没补。"""
    from core.qa import _all_texts
    fig, ax = new_figure("onehalf")
    ax.plot([1, 2], [1, 2], label="曲线")
    fig.legend(title="图级图例标题")
    assert any("图级图例标题" in t.get_text() for t in _all_texts(fig))


def test_probe_does_not_install_an_engine_on_a_plain_figure():
    """`set_layout_engine(None)` 会去读 rcParams：`figure.autolayout=True`
    时，一张本来没有 engine 的图恢复后会凭空拿到 TightLayoutEngine，
    紧接着的 draw 立即重排 —— 当场复现 N1 本身。"""
    import matplotlib as _mpl
    fig, ax = new_figure("onehalf")
    fig.set_layout_engine(None)
    ax.barh(["甲", "乙"], [0.6, 0.4], color="#08306b")
    with _mpl.rc_context({"figure.autolayout": True}):
        run_qa(fig, strict=False)
    assert fig.get_layout_engine() is None, fig.get_layout_engine()

def test_three_d_offset_text_is_inspected():
    from core.qa import _all_texts
    import matplotlib.pyplot as _plt
    fig = _plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot([1, 2], [1, 2], [1e7, 2e7])      # 大值要放在 z 上才触发
    ax.ticklabel_format(axis="z", style="sci", scilimits=(0, 0))
    fig.canvas.draw()
    assert any(t is ax.zaxis.get_offset_text() for t in _all_texts(fig))


# --- 第 8 轮 ---------------------------------------------------------

def test_minor_tick_labels_are_inspected():
    """上一轮我声称"这一支取不到任何东西、加不了测试"——三句全错。

    显式 minor formatter 下有 22 条非空标签，且**不用调 draw()**
    （`get_minorticklabels()` 内部会 `_update_ticks()` 现填内容）。
    这一支是带负载的：字号检查在这张图上当场命中 22 处。
    """
    from core.qa import _all_texts
    from matplotlib.ticker import FormatStrFormatter
    fig, ax = new_figure("onehalf")
    ax.plot([1, 6], [1, 2])
    ax.minorticks_on()
    ax.xaxis.set_minor_formatter(FormatStrFormatter("%.2f"))
    assert any(t.get_text().strip() == "1.20" for t in _all_texts(fig))


def test_minor_tick_labels_are_size_checked():
    from matplotlib.ticker import FormatStrFormatter
    fig, ax = new_figure("onehalf")
    ax.plot([1, 6], [1, 2])
    ax.minorticks_on()
    ax.xaxis.set_minor_formatter(FormatStrFormatter("%.2f"))
    ax.tick_params(axis="x", which="minor", labelsize=2.0)
    assert hit(fig, "字号")


def _dark_field(color, eff):
    fig, ax = new_figure("onehalf")
    ax.imshow(np.zeros((6, 6)), cmap="gray", vmin=0, vmax=1)
    t = ax.text(2.5, 2.5, "压在黑场上", color=color, fontsize=8, ha="center")
    t.set_path_effects(eff)
    return fig


def test_barely_visible_halo_alpha_does_not_grant_exemption():
    """上一轮只关了 lw 一半：`_rgba[3] <= 0.05` 只是二值门，过门后
    `_rgba[:3]` 把 alpha 丢掉，于是 6% 不透明度的描边按满不透明算。"""
    import matplotlib.patheffects as _pe
    fig = _dark_field("black", [_pe.withStroke(linewidth=3,
                                               foreground=(1, 1, 1, 0.06))])
    assert hit(fig, "对比度")


def test_zero_alpha_keyword_does_not_grant_exemption():
    """`_gc` 里的 alpha 键完全没读：`alpha=0.0` 时一个白像素都没画
    （像素实测 0 vs alpha=1.0 时 1118），硬拒检查照样被一个关键字关掉。"""
    import matplotlib.patheffects as _pe
    fig = _dark_field("black", [_pe.withStroke(linewidth=3,
                                               foreground="white", alpha=0.0)])
    assert hit(fig, "对比度")


def test_soft_halo_still_grants_exemption():
    """柔光晕（alpha 0.85 + 足够 lw）仍应放行，别过度修正。"""
    import matplotlib.patheffects as _pe
    fig = _dark_field("black", [_pe.withStroke(linewidth=2.4,
                                               foreground=(1, 1, 1, 0.85))])
    assert not hit(fig, "对比度")


def _perceptible_diff(a, b, thresh=8):
    """可感知像素数。

    反锯齿毛边会让"任意通道有差异"的计数虚高：实测场图 quad 接缝能贡献
    989 px，而中位通道差只有 2/255、Δ>8 的像素数为 0。用它当断言等于
    没有断言——第 9 轮那条"网格未渲染"的断言就是这样在网格确实空转时
    照样通过的。
    """
    import numpy as _np
    d = _np.abs(a[..., :3].astype(int) - b[..., :3].astype(int)).max(axis=2)
    return int((d > thresh).sum())


def _polar_demo():
    import sys as _sys
    import pathlib as _pl
    _rp = str(_pl.Path(__file__).resolve().parents[1] / "recipes")
    if _rp not in _sys.path:
        _sys.path.append(_rp)      # append 而非 insert：recipes/ 下有
    from contour_field import polar_field   # composition/parity 等通用名
    th = np.linspace(0, 2 * np.pi, 60)
    rr = np.linspace(0, 1, 30)
    T, R = np.meshgrid(th, rr)
    return polar_field(T, R, np.cos(3 * T) * R, zlabel="密度")


def _buf(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


def test_polar_field_renders_radial_tick_labels():
    """只隔离**径向刻度标签**本身。

    第 9 轮那版把 `set_yticks([])` 与藏 `ax.texts` 混在一起，于是在缺陷版
    上量到的 1141px 全部来自手写 `ax.text`、刻度贡献 0——测试绿着，缺陷
    还在。根因是本库样式的 `axes.axisbelow=True` 把极轴 zorder 压到 0.5，
    被 zorder=1 的场图整块盖住。
    """
    fig, ax = _polar_demo()
    base = _buf(fig)
    for lbl in ax.get_yticklabels():
        lbl.set_visible(False)
    assert _perceptible_diff(base, _buf(fig)) > 100, "径向刻度标签未渲染"
    plt.close(fig)


def test_polar_field_renders_radial_grid_rings():
    """只隔离**径向网格圆环**，且用可感知门槛。

    第 9 轮那条用 `ax.grid(False)` + "任意通道有差异"计数，在网格确实
    空转的两个变体上都得 989px 而通过——失败信息写着"ax.grid 空转"，
    却永远不会因为这个原因触发。
    """
    fig, ax = _polar_demo()
    base = _buf(fig)
    ax.yaxis.grid(False)
    assert _perceptible_diff(base, _buf(fig)) > 100, "径向网格圆环未渲染"
    plt.close(fig)


def test_polar_field_labels_track_a_non_zero_inner_radius():
    """`polar_field` 是公开 API。手写 `linspace(0, rmax, 6)` 无视 rmin：
    r∈[100,101] 时四个标签落在 20–81（低于 rmin、在轴外），读者会把半径
    读错两个数量级。"""
    import sys as _sys
    import pathlib as _pl
    _rp = str(_pl.Path(__file__).resolve().parents[1] / "recipes")
    if _rp not in _sys.path:
        _sys.path.append(_rp)
    from contour_field import polar_field
    th = np.linspace(0, 2 * np.pi, 40)
    rr = np.linspace(100, 101, 20)
    T, R = np.meshgrid(th, rr)
    fig, ax = polar_field(T, R, np.cos(3 * T) * R, zlabel="密度")
    fig.canvas.draw()
    lo, hi = ax.get_ylim()
    vals = list(ax.get_yticks())
    for t in ax.texts:
        try:
            vals.append(float(t.get_text()))
        except ValueError:
            pass
    assert [v for v in vals if lo <= v <= hi],         f"没有径向标注落在轴内 {lo:.4g}–{hi:.4g}：{vals}"
    plt.close(fig)


def test_opaque_halo_with_translucent_foreground_is_exempt():
    """mpl 的 alpha 语义是**覆盖**不是相乘：`set_alpha` 会置 `_forced_alpha`
    并重刷 `_rgb`，所以 `foreground=(1,1,1,0.06) + alpha=1.0` 画出的是
    **纯白**光晕（像素实测 1920 个白点）。写成相乘会把它当成 6% 而误报。

    这是第 9 轮 commit 自己点名、却没写进测试的那个判别用例。
    """
    import matplotlib.patheffects as _pe
    fig, ax = new_figure("onehalf")
    ax.imshow(np.zeros((6, 6)), cmap="gray", vmin=0, vmax=1)
    t = ax.text(2.5, 2.5, "压在黑场上", color="black", fontsize=8, ha="center")
    t.set_path_effects([_pe.withStroke(linewidth=3,
                                       foreground=(1, 1, 1, 0.06),
                                       alpha=1.0)])
    assert not hit(fig, "对比度")


# --- 第 11 轮：用户在正确用法上撞墙 -----------------------------------

def test_contourf_counts_as_a_field_not_a_one_dimensional_panel():
    """`FIELD_CLASSES` 漏了 `QuadContourSet`（contourf 在 mpl≥3.8 的产物），
    于是同一张图 pcolormesh 过、contourf 被判"4/4 面板是一维构图"硬拒，
    并被指向 contour_field.py——正是唯一用 contourf 的那个 recipe。"""
    from core import cmap_for
    from core import _probe
    X, Y = np.meshgrid(np.linspace(0, 1, 30), np.linspace(0, 1, 30))
    Z = np.sin(3 * X) * np.cos(3 * Y)
    fig, ax = new_figure("onehalf")
    ax.contourf(X, Y, Z, levels=10, cmap=cmap_for("sequential"))
    assert _probe.has_field(ax), "contourf 未被识别为场图"
    assert not hit(fig, "一维构图")


def test_stat_box_auto_falls_back_outside_on_a_contourf_field():
    """`stat_box` 的 docstring 与 api.md 都承诺"满铺场图（imshow /
    pcolormesh / **contourf**）…loc='auto' 会自动降级到 outside='top'"。
    降级条件是 `_probe.has_field(ax)`，contourf 不在其列 → 承诺落空、
    框照压场图，随即被遮挡检查硬拒。"""
    from core import cmap_for
    X, Y = np.meshgrid(np.linspace(0, 1, 30), np.linspace(0, 1, 30))
    fig, ax = new_figure("onehalf")
    ax.contourf(X, Y, np.sin(3 * X) * np.cos(3 * Y), levels=10,
                cmap=cmap_for("sequential"))
    stat_box(ax, ["n = 900", "RMS = 0.41"], loc="auto")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.suptitle("场在 x=0.5 处达峰")
    assert not hit(fig, "压在场图上")


def test_share_colorbar_keeps_the_declared_column_width():
    """`share_colorbar(loc="right", label=…)` 使交付宽 190mm（目标 183），
    无论传不传 expect_width 都硬拒——而 `figure()` + `share_colorbar`
    正是 SKILL.md §4b 主推的两个 API。"""
    from core import figure, share_colorbar, cmap_for
    from core.style import delivered_width_in
    X, Y = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20))
    fig, axd = figure([[("a", 1), ("b", 1)]], width="double", height=100)
    im = axd["a"].pcolormesh(X, Y, np.sin(3 * X), cmap=cmap_for("sequential"),
                             shading="auto")
    share_colorbar(fig, im, list(axd.values()), label="相对密度")
    w = delivered_width_in(fig) * 25.4
    assert abs(w - 183) <= 3.0, f"交付宽 {w:.1f}mm 不在 183±3"


def test_cross_axes_tick_and_label_overlap_is_flagged():
    """跨轴的"刻度 vs 排版文字"重叠此前无人覆盖：5b3 排除了刻度、
    6b 只比同一根轴内的相邻刻度。

    实测：`share_colorbar` 挂在双面板的左格上时，色标刻度与右格的
    ylabel 重叠 21%，两者水平间距 2px，目视糊成一团，而 QA 全绿。
    """
    from core import figure, share_colorbar, cmap_for
    X, Y = np.meshgrid(np.linspace(0, 1, 30), np.linspace(0, 1, 30))
    fig, ax = figure([[("a", 1), ("b", 1)]], width="double", height=100)
    im = ax["a"].pcolormesh(X, Y, np.sin(3 * X),
                            cmap=cmap_for("sequential"), shading="auto")
    ax["b"].plot([1, 2], [1, 2])
    ax["b"].set_ylabel("总成本 / 元")
    share_colorbar(fig, im, [ax["a"]], label="相对密度")
    assert hit(fig, "重叠")


def test_ticks_of_the_same_axis_do_not_trigger_the_cross_axes_check():
    """同一根轴内的刻度密集不该由这条检查报（6b 专管，且判据不同）。"""
    fig, ax = new_figure("onehalf")
    ax.plot(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    assert not hit(fig, "跨面板文字重叠")


def test_clipped_out_of_view_ticks_do_not_trigger_the_cross_axes_check():
    """matplotlib 保留**视窗外**刻度的 artist：它们渲染时被裁掉、肉眼
    不可见，但 `get_window_extent()` 仍报位置，而那位置常落在邻格上。

    实测 facet_metrics 的 log 面板：首个刻度报在 x=330，而它自己的轴是
    441–667，于是与左邻格的刻度"重叠 100%"。判据必须排掉这些。
    """
    import sys as _sys
    import pathlib as _pl
    _rp = str(_pl.Path(__file__).resolve().parents[1] / "recipes")
    if _rp not in _sys.path:
        _sys.path.append(_rp)
    from comparison_rank import facet_metrics
    fig, axes = facet_metrics(
        ["A", "B", "C", "D", "E"],
        [("总成本（元）", [9.13, 9.6, 10.2, 11.0, 12.4], "linear"),
         ("失效率", [0.002, 0.02, 0.2, 2.0, 20.0], "log"),
         ("满意度", [4.6, 4.2, 3.9, 3.1, 2.8], "linear")])
    assert not hit(fig, "跨面板文字重叠")


# --- 第 12 轮 ---------------------------------------------------------

def test_line_contours_are_not_treated_as_a_filled_field():
    """`contour()` 与 `contourf()` 在 mpl≥3.8 同为 `QuadContourSet`，
    上一轮把整个类塞进 FIELD_CLASSES，于是**线**等值线也被当成满铺场：
    一张 92% 是白底的合规等值线图，只因放了个图例就被判"图例压在场图上
    （框底 100% 是色块）"——诊断与事实相反，而补救（改 stat_box
    outside）对图例根本不适用。两者可用 `filled` 属性区分。
    """
    from core import _probe
    X, Y = np.meshgrid(np.linspace(0, 1, 40), np.linspace(0, 1, 40))
    Z = np.sin(3 * X) * np.cos(3 * Y)
    fig, ax = new_figure("onehalf")
    ax.contour(X, Y, Z, levels=8, colors=PALETTE[0])
    assert not _probe.has_field(ax), "线等值线不该算满铺场"

    fig2, ax2 = new_figure("onehalf")
    ax2.contourf(X, Y, Z, levels=8, cmap="YlOrRd")
    from core import _probe as _p2
    assert _p2.has_field(ax2), "填充等值线应算场图"


def test_a_legend_on_a_line_contour_plot_is_not_rejected():
    """端到端：合规的线等值线图 + 图例，不该被硬拒。"""
    X, Y = np.meshgrid(np.linspace(0, 1, 40), np.linspace(0, 1, 40))
    Z = np.sin(3 * X) * np.cos(3 * Y)
    fig, ax = new_figure("onehalf")
    cs = ax.contour(X, Y, Z, levels=8, colors=PALETTE[0])
    ax.clabel(cs, inline=True, fontsize=6.5)
    ax.plot([0, 1], [0.2, 0.8], color=PALETTE[1], label="约束前沿")
    ax.legend(loc="upper left")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.suptitle("前沿切点给出最低成本解")
    stat_box(ax, ["n = 1600"], loc="lower right")
    assert not hit(fig, "压在场图上")


def test_end_labels_actually_separate_by_the_requested_point_gap():
    """`end_labels` 把**显示像素**与**点**混用：`min_gap_pt=9` 在
    figure.dpi=150 下实际只拉开 9×72/150 = 4.3pt，而标签本身高 7.6pt——
    末端接近的两条线必然叠字，还会被自家的直标互压检查拦下。
    而 api.md 承诺的是"一次排完并自动避让"。
    """
    from core import end_labels
    fig, ax = new_figure("onehalf")
    x = np.linspace(0, 1, 20)
    ax.plot(x, 0.50 + 0.0 * x, color=PALETTE[0])
    ax.plot(x, 0.505 + 0.0 * x, color=PALETTE[1])
    end_labels(ax, [(1.0, 0.500, "方案A", PALETTE[0]),
                    (1.0, 0.505, "方案B", PALETTE[1])])
    assert not hit(fig, "直标互相重叠")


def test_callout_on_a_dark_field_is_not_rejected():
    """`callout` 在深色场上主动选择半透明白底框（白描边在深底读不清），
    上一轮给它打了 `_ff_intentional_box`，但那只挡 5c；实际开火的是 5d
    像素兜底，它完全不看这个标记 → 照 §4 骨架写 callout 打在深场上必被
    硬拒，而引线注释必须锚在数据点上，没有"挪到轴外"这个选项。"""
    X, Y = np.meshgrid(np.linspace(0, 1, 40), np.linspace(0, 1, 40))
    fig, ax = new_figure("onehalf")
    ax.pcolormesh(X, Y, X * Y, cmap="viridis", shading="auto")
    callout(ax, xy=(0.5, 0.5), text="峰值点")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.suptitle("场在中心达峰")
    assert not hit(fig, "真空区")


def test_focus_reference_line_does_not_share_a_colour_with_series_two():
    """`ref_line(level="focus")` 默认色 == `semantic("highlight")` ==
    `PALETTE[1]` == `categorical(n)[1]`：两条系列以上再加一条焦点判据线，
    判据线必与第二条系列同色，违反"配色语义一致"。"""
    from core import semantic, categorical
    cols, _, _ = categorical(3)
    assert semantic("highlight").upper() != cols[1].upper(), \
        f"判据线色 {semantic('highlight')} 与第二条系列同色"


def test_horizontal_bars_are_sampled_at_full_length():
    """`series_samples` 对 `ax.containers` 一律按**竖柱**采样
    `[(x0+x1)/2, y1]`，barh 因此只上报一半条长（实测 0.04/0.07/0.09
    vs 真值 0.08/0.14/0.18）——而横条正是硬拒绝清单第一条推荐的替代构图，
    "图例压柱顶"检查对它等于空转。"""
    from core import _probe
    vals = [0.08, 0.14, 0.18]
    fig, ax = new_figure("onehalf")
    ax.barh([0, 1, 2], vals, color=PALETTE[0])
    fig.canvas.draw()
    got = set()
    for _nm, pts in _probe.series_samples(ax):
        d = ax.transData.inverted().transform(pts)
        got.update(np.round(d[:, 0], 3))
    for v in vals:
        assert any(abs(g - v) < 0.005 for g in got), \
            f"条长 {v} 未被采到；采到 {sorted(got)}"


def test_expand_axes_is_not_a_dead_parameter():
    """`expand_axes` 在签名、docstring、api.md 三处登记，函数体里从未读取。
    api.md 明写"最佳位置仍压数据时，按框高扩一档轴限腾出真空带，而不是
    压上去"——实测多面板里轴外放不下时会**静默压回轴内**等着被 QA 拦。
    """
    import inspect
    from core import stat_box as _sb
    src = inspect.getsource(_sb)
    body = src.split('"""', 2)[-1]
    assert "expand_axes" in body, "expand_axes 在函数体里从未被读取"


def test_stat_box_expands_limits_instead_of_covering_data():
    """满格数据 + 轴外放不下 → 应扩轴限腾出真空带，而不是压在数据上。"""
    import matplotlib.pyplot as _plt
    fig, axes = _plt.subplots(2, 1, figsize=(3.5, 3.0))
    rng = np.random.default_rng(0)
    for a in axes:
        a.scatter(rng.random(400), rng.random(400), s=4, color=PALETTE[0])
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
    y0, y1 = axes[0].get_ylim()
    stat_box(axes[0], ["n = 400", "RMS = 0.29"], loc="auto")
    fig.canvas.draw()
    ny0, ny1 = axes[0].get_ylim()
    moved_outside = any(t.get_position()[1] > 1.0 or t.get_position()[1] < 0
                        for t in axes[0].texts)
    assert (ny1 - ny0) > (y1 - y0) + 1e-9 or moved_outside, \
        f"轴限未扩（{y0},{y1} → {ny0},{ny1}）且未移到轴外"


def test_fill_between_bands_are_visible_to_occupancy_probing():
    """`fill_between` 在 mpl 3.10 返回 `FillBetweenPolyCollection`，而它的
    `get_offsets()` 返回退化的 `[[0,0]]`（size=2）——于是 offsets 分支
    **抢先命中**，整片置信带只产出一个 (0,0) 幽灵点，顶点采样那一支永远
    走不到。

    后果正是源码注释预言的那句：占用栅格说"这里空"、像素实测说"41% 有
    内容"，于是 `loc="auto"` 主动把框放到带上、再被自家像素兜底硬拒，
    而消息只说"该位置不是真空区"，没有可操作的下一步。

    置信带是本库主推构图（convergence_ci / timeseries_forecast /
    raincloud / joint_marginal 全靠它）。
    """
    from core import _probe
    fig, ax = new_figure("onehalf")
    x = np.linspace(0, 1, 40)
    y = np.sin(6 * x)
    ax.fill_between(x, y - 0.3, y + 0.3, alpha=0.3, color="#AECDE1")
    ax.plot(x, y, color=PALETTE[0])
    fig.canvas.draw()
    band = [(n, p) for n, p in _probe.series_samples(ax) if n != "line0"]
    assert band, "置信带完全没被采到"
    n_pts = max(len(p) for _, p in band)
    assert n_pts > 20, f"置信带只采到 {n_pts} 个点（幽灵点）"


def test_auto_placement_does_not_land_on_a_confidence_band():
    """端到端：宽置信带上 `loc="auto"` 不该把框放上去再被自家硬拒。"""
    fig, ax = new_figure("onehalf")
    x = np.linspace(0, 1, 60)
    y = 0.5 + 0.35 * np.sin(6 * x)
    ax.fill_between(x, y - 0.32, y + 0.32, alpha=0.35, color="#AECDE1")
    ax.plot(x, y, color=PALETTE[0])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim(0, 1)
    fig.suptitle("带宽随 x 收敛")
    stat_box(ax, ["n = 60", "半宽 0.32"], loc="auto")
    assert not hit(fig, "真空区")


def test_reference_line_focus_colour_follows_semantic_highlight():
    """`ref_line(level="focus")` 硬编码 `#D55E00`，从不读
    `semantic("highlight")`——于是换色只改了 `dot_interval` 一处，库里
    出现两个互不相同的"判据线色"（fit_residual.png 同图内即可见：阈值线
    橙红、其交点星标粉红）。"""
    from core import semantic, ref_line as _rl
    fig, ax = new_figure("onehalf")
    ax.plot([0, 1], [0, 1], color=PALETTE[0])
    _rl(ax, 0.5, orientation="h", label="判据", level="focus")
    cols = {ln.get_color().upper() for ln in ax.lines
            if str(ln.get_linestyle()) not in ("-", "None")}
    assert semantic("highlight").upper() in cols, \
        f"判据线色 {cols} 未跟随 semantic('highlight')"


def test_ptx_is_identity_under_cn_and_passes_zero_through():
    """`ptx` 在 cn 档必须恒等（否则 245 处机械替换会悄悄改变既有版面），
    且 0 要原样透传——`lw=0`（无边框填充）与"不画文字"都是合法输入，
    把它钳到字号下限是错的。"""
    from core import apply_style as _as, ptx
    _as("cn")
    for v in (6.5, 7, 7.5, 8, 9, 10.5):
        assert ptx(v) == v, f"cn 档 ptx({v}) = {ptx(v)}"
    for v in (0.3, 0.6, 0.8, 1.0, 1.2):
        assert abs(ptx(v, "lw") - v) < 1e-9, f"cn 档 ptx({v},'lw') = {ptx(v,'lw')}"
    assert ptx(0) == 0, f"ptx(0) = {ptx(0)}，0 应原样透传"
    assert ptx(0, "lw") == 0


def test_ptx_scales_into_the_nature_envelope():
    """nature 档：cn 调好的值要落进该档的字号/线宽包线内。"""
    from core import apply_style as _as, ptx
    _as("nature")
    try:
        for v in (9, 10.5, 12):
            assert 5.0 <= ptx(v) <= 7.0, f"ptx({v}) = {ptx(v)} 超出 nature 包线"
        for v in (1.2, 1.6, 2.4):
            assert ptx(v, "lw") <= 1.0, f"ptx({v},'lw') = {ptx(v,'lw')} > 1pt"
        assert ptx(0) == 0
    finally:
        _as("cn")


def test_ptx_lw_is_identity_under_cn():
    """`min(out, cfg["line_width"])` 把 cn 的**默认**线宽 1.2 当成了**上限**，
    于是 22 处调用点在 cn 下被白白压细——而 cn 档根本没有线宽 QA 检查。

    更要紧的是它与库内既有的 `_mark_lw` 直接打架：后者明确写着"森林图的
    粗横棒、斜率图的高亮线在 cn 档下本该有分量"，在 cn 恒等。同一个库里
    两套互相矛盾的线宽政策。
    """
    from core import apply_style as _as, ptx
    from core.annotate import _mark_lw
    _as("cn")
    for v in (1.3, 1.6, 2.0, 2.4, 3.0):
        assert abs(ptx(v, "lw") - v) < 1e-9, f"cn 档 ptx({v},'lw') = {ptx(v,'lw')}"
        assert abs(ptx(v, "lw") - _mark_lw(v)) < 1e-9, "与 _mark_lw 政策不一致"


def test_ptx_lw_respects_the_nature_ceiling():
    from core import apply_style as _as, ptx
    _as("nature")
    try:
        for v in (1.3, 1.6, 2.0, 3.0):
            assert ptx(v, "lw") <= 1.0 + 1e-9
    finally:
        _as("cn")


def test_grid_off_by_preset_actually_turns_the_grid_off():
    """`ax.grid(False, linewidth=…, alpha=…)` 在 matplotlib 里会**反向生效**
    ——它当场警告 "First parameter to grid() is false, but line properties
    are supplied. The grid will be enabled."，网格反而被打开。"""
    import sys as _s
    import pathlib as _pl
    _rp = str(_pl.Path(__file__).resolve().parents[1] / "recipes")
    if _rp not in _s.path:
        _s.path.append(_rp)
    from core import apply_style as _as
    from contour_field import polar_field
    _as("nature")
    try:
        th = np.linspace(0, 2 * np.pi, 40)
        rr = np.linspace(0, 1, 20)
        T, R = np.meshgrid(th, rr)
        fig, ax = polar_field(T, R, np.cos(3 * T) * R, zlabel="密度")
        fig.canvas.draw()
        on = [t.gridline.get_visible() for t in ax.yaxis.get_major_ticks()]
        assert not any(on), "nature 档下极坐标网格仍开着"
        plt.close(fig)
    finally:
        _as("cn")


def test_a_copied_recipe_runs_after_deleting_the_common_import():
    """SKILL.md 工作流第 5 步是"从 recipes 复制模板"，并明确指示"复制到
    自己的脚本时删掉 `from _common import …` 这一行"。

    而 recipe 顶部一度是 `from _common import GALLERY, PRESET` +
    `apply_style(PRESET)`——照文档删掉那行就 `NameError: PRESET`，
    24/24 个模板全中。档位选择应该在库里（apply_style 认 FF_PRESET），
    不该寄生在 demo 专用的路径工具上。
    """
    import subprocess
    import sys as _s
    import pathlib as _pl
    import tempfile
    import os
    root = _pl.Path(__file__).resolve().parents[1]
    src = (root / "recipes" / "scan_curve.py").read_text(encoding="utf-8")
    body = "\n".join(l for l in src.split("\n")
                     if not l.startswith("from _common import"))
    with tempfile.TemporaryDirectory() as d:
        out = _pl.Path(d) / "copied.py"
        head = (f"import sys; sys.path.insert(0, r'{root}')\n"
                f"import pathlib; GALLERY = pathlib.Path(r'{d}')\n")
        out.write_text(head + body, encoding="utf-8")
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        r = subprocess.run([_s.executable, str(out)], capture_output=True,
                           text=True, encoding="utf-8", errors="replace",
                           env=env, cwd=d)
    assert r.returncode == 0, f"复制后的脚本跑不起来：\n{r.stderr[-600:]}"


def test_ref_line_rejects_an_unknown_orientation():
    """`orientation` 只认字面量 "h"，**任何其他值都静默落进竖线分支**，
    而 docstring 与 api.md 都只写了默认值 "h"、从没说"v"是唯一另一个合法值。

    `orientation="horizontal"` 是极自然的写法（尤其 LLM 按语义直觉写），
    结果画成竖线、阈值点错轴，而 run_qa 全部 30 条检查无一命中——这是
    唯一一条会让用户拿到**几何错误**的图却照常交付的缺陷。
    """
    from core import ref_line as _rl
    fig, ax = new_figure("onehalf")
    ax.plot([0, 10], [0, 10])
    with pytest.raises(ValueError):
        _rl(ax, 5, orientation="horizontal", label="阈值")


def test_ref_line_h_and_v_still_work():
    from core import ref_line as _rl
    fig, ax = new_figure("onehalf")
    ax.plot([0, 10], [0, 10])
    _rl(ax, 5, orientation="h")
    _rl(ax, 3, orientation="v")
    xs = [tuple(np.round(ln.get_xdata()[:2], 6)) for ln in ax.lines[1:]]
    assert xs[0][0] != xs[0][1], "h 应画横线"
    assert xs[1][0] == xs[1][1], "v 应画竖线"


def test_sparse_panel_ink_check_is_preset_aware():
    """墨迹密度的两条硬检查既不按 `current_preset()` 分档、也不在
    `_ALLOW_CODES` 里（直接 `problems.append`，没有 allow 出口）。

    nature 档字号 ≤7pt、线宽 ≤1pt，同一构图墨迹必然更低，而阈值没跟着降：
    algo_convergence 7.5%→4.2%、fit_residual 8.1%→4.0%，于是 cn 下只是
    WARN 的图在 nature 下变成不可豁免的硬拒。
    """
    from core.qa import _ALLOW_CODES
    assert "sparse_panel" in _ALLOW_CODES, "墨迹检查没有 allow 出口"


def test_sparse_panel_can_be_waived():
    fig, ax = new_figure("onehalf")
    ax.plot([1, 2, 3], [1, 2, 3], color=PALETTE[0])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.suptitle("结论句")
    stat_box(ax, ["n = 3"], loc="auto")
    assert not any("墨迹" in p for p in
                   run_qa(fig, strict=False, allow=("sparse_panel",)))


# --- R16 公开 API 的枚举参数：非法值必须报错，不能静默走另一分支 -------
#
# 第 15 轮修了 `ref_line(orientation=)`，但那是**一类**缺陷的一个实例。
# 第 16 轮三方评审把同一模式又找出五处：字符串枚举参数没有校验、
# 非法值落进 `else` 兜底分支，用户拿到的图与他写的代码不符而毫无提示。
# 这组测试按「表驱动」写：新增枚举参数时把它加进表里，漏校验立刻变红。

_ENUM_CASES = [
    # (说明, 构造并调用的函数, 非法值)
    ("dot_interval(better=)", "dot_interval", "higher"),
    ("dot_interval(better=)", "dot_interval", "HIGH"),
    ("ref_line(label_loc=) 横线", "ref_line_h", "top"),
    ("ref_line(label_loc=) 竖线", "ref_line_v", "left"),
    ("stat_box(loc=)", "stat_box_loc", "top left"),
    ("stat_box(outside=)", "stat_box_outside", "above"),
    ("slope_lines(mode=)", "slope_lines_mode", "cohorts"),
    ("ptx(kind=)", "ptx_kind", "linewidth"),
]


def _call_with(which, value):
    """按 which 调用对应公开 API，把 value 塞进那个枚举参数。"""
    from core import (dot_interval as _di, ref_line as _rl,
                      stat_box as _sb, slope_lines as _sl)
    from core.style import ptx as _ptx
    if which == "ptx_kind":
        return _ptx(1.6, value)
    fig, ax = new_figure("onehalf")
    ax.plot([0, 10], [0, 10])
    if which == "dot_interval":
        return _di(ax, ["a", "b"], [0.7, 0.3], [0.6, 0.2], [0.8, 0.4],
                   threshold=0.5, better=value, sort=False, value_col=False)
    if which == "ref_line_h":
        return _rl(ax, 5, orientation="h", label="阈值", label_loc=value)
    if which == "ref_line_v":
        return _rl(ax, 5, orientation="v", label="阈值", label_loc=value)
    if which == "stat_box_loc":
        return _sb(ax, ["n = 3"], loc=value)
    if which == "stat_box_outside":
        return _sb(ax, ["n = 3"], outside=value)
    if which == "slope_lines_mode":
        return _sl(ax, ["a", "b"], [1.0, 2.0], [2.0, 1.0], mode=value)
    raise AssertionError(which)


@pytest.mark.parametrize("desc,which,bad", _ENUM_CASES,
                         ids=[f"{c[0]}={c[2]}" for c in _ENUM_CASES])
def test_enum_parameters_reject_unknown_values(desc, which, bad):
    """非法枚举值必须抛 ValueError，且错误信息里要列出合法值。

    「静默走另一分支」比报错伤得多：报错用户当场改，静默错的图会直接
    进论文。错误信息不列合法值也不行——用户只能去读源码。
    """
    with pytest.raises(ValueError) as e:
        _call_with(which, bad)
    assert bad in str(e.value), f"{desc}：错误信息没回显收到的非法值"


def test_dot_interval_better_high_and_low_are_opposite():
    """`better` 的两个合法值必须给出相反判定——这是「该触发/不该触发」
    的另一侧：光校验非法值，万一把 high 也一起改坏就没人拦得住。
    """
    from core import dot_interval as _di
    fig, ax = new_figure("onehalf")
    hi = list(map(bool, _di(ax, ["a", "b"], [0.7, 0.3], [0.6, 0.2],
                            [0.8, 0.4], threshold=0.5, better="high",
                            sort=False, value_col=False)))
    fig2, ax2 = new_figure("onehalf")
    lo = list(map(bool, _di(ax2, ["a", "b"], [0.7, 0.3], [0.6, 0.2],
                            [0.8, 0.4], threshold=0.5, better="low",
                            sort=False, value_col=False)))
    assert hi == [True, False], f"better='high' 判定错了：{hi}"
    assert lo == [False, True], f"better='low' 判定错了：{lo}"


def test_ref_line_label_loc_actually_moves_the_label():
    """竖线分支根本不读 `label_loc`——它是个死参数：文档登记了、
    函数体不读。两个合法值必须落在不同位置，否则参数等于不存在。
    """
    from core import ref_line as _rl
    pos = {}
    for loc in ("top", "bottom"):
        fig, ax = new_figure("onehalf")
        ax.plot([0, 10], [0, 10])
        _rl(ax, 5, orientation="v", label="阈值", label_loc=loc)
        pos[loc] = tuple(round(v, 4) for v in ax.texts[-1].get_position())
    assert pos["top"] != pos["bottom"], f"竖线 label_loc 是死参数：{pos}"


def test_ptx_lw_typo_does_not_return_a_font_sized_number():
    """`ptx(1.6, "linewidth")` 落进 font 分支后被字号**下限**兜成 5.0——
    用户要的是 1.0pt 线宽，拿到 5.0pt，整整 5 倍且毫无提示。
    """
    from core.style import ptx as _ptx
    apply_style("nature")
    try:
        assert _ptx(1.6, "lw") <= 1.0
        with pytest.raises(ValueError):
            _ptx(1.6, "linewidth")
    finally:
        apply_style("cn")


# --- R16 N-5：非有限数值不能静默变成图上的 nan / 退化标注 --------------

def test_callout_rejects_non_finite_target():
    """NaN 目标坐标的 callout 渲染成 1×1 退化框，肉眼完全不可见，
    而 run_qa 报 PASS——用户以为标注上了，交付的图上没有。
    """
    from core import callout as _co
    fig, ax = new_figure("onehalf")
    ax.plot([0, 1], [0, 1])
    with pytest.raises(ValueError):
        _co(ax, (float("nan"), 0.5), "峰值", (0.7, 0.3))


def test_callout_finite_target_still_works():
    from core import callout as _co
    fig, ax = new_figure("onehalf")
    ax.plot([0, 1], [0, 1])
    a = _co(ax, (0.5, 0.5), "峰值", (0.7, 0.3))
    assert a is not None


def test_qa_rejects_nan_rendered_into_a_title():
    """图题里出现 `nan` / `inf` 一律是上游算错了还漏了检查——
    这种图不该落盘。
    """
    fig, ax = new_figure("onehalf")
    ax.plot([1, 2, 3], [1, 2, 3])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.suptitle(f"最优值 = {float('nan')}")
    assert hit(fig, "nan")


def test_qa_rejects_an_array_repr_rendered_into_a_title():
    fig, ax = new_figure("onehalf")
    ax.plot([1, 2, 3], [1, 2, 3])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.suptitle(f"候选 = {np.array([1.0, 2.0])}")
    assert hit(fig, "数组")


def test_qa_does_not_flag_an_ordinary_title():
    """不该触发的一侧：正常图题里带小数、单位、百分号都不能误报。"""
    fig, ax = new_figure("onehalf")
    ax.plot([1, 2, 3], [1, 2, 3])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.suptitle("最优值 = 5.07 cm，覆盖率 94.2%")
    assert not hit(fig, "nan")
    assert not hit(fig, "数组")


# --- R16b 同一类的余下三处：未知宽度档 / value_col ----------------------

def test_layout_width_rejects_an_unknown_column_name():
    """`_width_mm` 对未知字符串原样返回，于是 `figure(width="triple")` 一路
    走到 `"triple" * 0.0393` 才炸出 `TypeError: can't multiply sequence`——
    能拦住，但错误信息里没有半点线索指向 width。`new_figure` 早就校验了，
    同一个库里不能有两套政策。
    """
    from core.layout import figure as _fig
    with pytest.raises(ValueError) as e:
        _fig({"a": "x"}, width="triple")
    assert "triple" in str(e.value)


def test_run_qa_rejects_an_unknown_expect_width():
    """`expect_width="triple"` 同样走 `COLUMN_WIDTHS.get(w, w)` 兜底，
    最后炸在 numpy 的 `UFuncTypeError` 上，用户完全无从下手。
    """
    fig, ax = new_figure("single")
    ax.plot([0, 1], [0, 1])
    with pytest.raises(ValueError) as e:
        run_qa(fig, expect_width="triple", strict=False)
    assert "triple" in str(e.value)


def test_run_qa_still_takes_a_mm_number_and_a_tuple():
    """不该触发的一侧：mm 数值与档名元组都要照旧生效。"""
    fig, ax = new_figure("single")
    ax.plot([0, 1], [0, 1])
    run_qa(fig, expect_width=89, strict=False)
    run_qa(fig, expect_width=("single", "onehalf"), strict=False)


def test_dot_interval_value_col_rejects_an_unknown_string():
    """`value_col` 只认 True / False / "inside"，但实现只判 truthy：
    `value_col="False"`（字符串）是 truthy，用户想关掉数值列，结果照画。
    """
    from core import dot_interval as _di
    fig, ax = new_figure("onehalf")
    with pytest.raises(ValueError):
        _di(ax, ["a", "b"], [0.7, 0.3], [0.6, 0.2], [0.8, 0.4],
            threshold=0.5, better="high", sort=False, value_col="False")


def test_dot_interval_value_col_three_legal_values_differ():
    """不该触发的一侧：三个合法值必须各走各的分支。"""
    from core import dot_interval as _di
    n = {}
    for vc in (True, False, "inside"):
        fig, ax = new_figure("onehalf")
        _di(ax, ["a", "b"], [0.7, 0.3], [0.6, 0.2], [0.8, 0.4],
            threshold=0.5, better="high", sort=False, value_col=vc)
        n[str(vc)] = len(ax.texts)
    assert len(set(n.values())) == 3, f"三个合法值没走出三种行为：{n}"


# --- R16c SKILL.md §5 教的写法必须在两档下都真的能过 --------------------

def test_skill_md_section5_advice_survives_the_nature_preset():
    """入口文档 §5 此前教「裸 `ax.annotate(color=…)` 自己套 `ink(color)`」。
    `ink()` 只压暗、不除彩，而 nature 档「彩色文字」是 `problems.append`
    的硬拒且没有 allow 出口——照入口文档写，在 nature 档 100% 被拦下，
    这是「正确用法上被无理由硬拒」。§5 已改教 `text_color()`，这条钉住它。
    """
    from core import ink as _ink, text_color as _tc, semantic as _sem
    try:
        apply_style("nature")
        col = _sem("highlight")
        # 文档教的写法：必须不产生彩色文字
        assert not _qa_flags_coloured_text(_tc(col)), \
            "§5 教的 text_color() 在 nature 档仍被判彩色文字"
        # 反面：旧写法确实会被拦——证明这条硬拒是真的，不是我编的
        assert _qa_flags_coloured_text(_ink(col)), \
            "ink() 在 nature 档没被拦，那 §5 的改动就是无的放矢"
    finally:
        apply_style("cn")


def _qa_flags_coloured_text(color):
    fig, ax = new_figure("onehalf")
    ax.plot([0, 1], [0, 1])
    ax.annotate("关键点", xy=(0.5, 0.5), xytext=(0.6, 0.3), color=color)
    out = any("彩色文字" in p for p in run_qa(fig, strict=False))
    plt.close(fig)
    return out


def test_skill_md_section5_advice_also_holds_under_cn():
    """cn 档允许彩色文字，但 text_color() 仍要保证 ≥3:1 对比度。"""
    from core import text_color as _tc, semantic as _sem
    from core.colors import contrast_ratio
    apply_style("cn")
    assert contrast_ratio(_tc(_sem("highlight")), "white") >= 3.0


# --- R16d 随库交付的模板必须在两个交付档下都能过 ------------------------


def _recipe(name):
    """按 recipe 自己的运行方式导入：它们 `from _common import GALLERY`，
    要求 `recipes/` 在 sys.path 上（`python recipes/xxx.py` 时自动成立）。
    """
    import importlib
    rd = str(pathlib.Path(__file__).resolve().parents[1] / "recipes")
    if rd not in sys.path:
        sys.path.insert(0, rd)
    return importlib.import_module(name)


def test_facet_metrics_template_passes_under_the_nature_preset():
    """`facet_metrics` 是随库交付的模板，SKILL.md 的工作流要求"从 recipes
    抄构图"。它在 nature 档被墨迹密度硬拒（18cm² 面板 2.5%），等于用户
    照文档走主路径就撞墙——这是「正确用法上被无理由硬拒」。

    修法不是加豁免：nature 档字号 7pt、线宽 ≤1pt、标记更小，同一构图的
    墨迹必然更低，**面板本来就该按内容变矮**。面板高度跟着档走，密度自然
    回到阈值以上，cn 档一个像素都不动。
    """
    try:
        apply_style("nature")
        cr = _recipe("comparison_rank")
        fig, axes = cr.facet_metrics(
            ["分支定界", "割平面", "禁忌搜索", "模拟退火", "贪心"],
            [("求解时间（s）", [312.0, 96.4, 21.7, 4.3, 0.8], "log"),
             ("峰值内存（MB）", [1850.0, 940.0, 410.0, 180.0, 95.0], "log"),
             ("最优性 gap（%）", [0.0, 0.3, 0.9, 1.6, 2.4], "linear")])
        fig.suptitle("结论句")
        bad = [p for p in run_qa(fig, expect_width=("double",), strict=False)
               if "墨迹" in p]
        assert not bad, f"nature 档模板被墨迹硬拒：{bad}"
    finally:
        apply_style("cn")


def test_facet_metrics_keeps_its_cn_geometry():
    """不该触发的一侧：cn 档的图形尺寸不能因为上面的修法发生任何变化
    （gallery 里 25 张交付图都是 cn 档出的，动了就是回归）。
    """
    apply_style("cn")
    cr = _recipe("comparison_rank")
    fig, axes = cr.facet_metrics(
        ["a", "b", "c"], [("m1", [1.0, 2.0, 3.0], "linear")])
    from core import COLUMN_WIDTHS, MM
    w_in = COLUMN_WIDTHS["double"] * MM
    assert abs(fig.get_size_inches()[0] - w_in) < 1e-9
    assert abs(fig.get_size_inches()[1] - w_in * 0.36) < 1e-9


# --- R16e recipes 层的同一类缺陷 ---------------------------------------
#
# 三位评审都只查了 `core/`，但 `recipes/` 的函数同样是交付给用户的公开
# API（api.md 逐个登记了签名）。往这层一扫又出两条，其中 parity 的那条
# 比 core 里任何一条都重：拼错一个字母，写进论文的覆盖率从 98% 变成 20%。

def test_parity_rejects_an_unknown_band_kind():
    """`band=("relatve", 0.10)` 静默落进绝对带分支：δ 被当成**绝对**单位
    而不是 10% 相对，`info["inside"]` 从 0.9833 掉到 0.2——而这个数会被
    写进图题和统计框，直接进论文。全库最重的一条静默走错分支。
    """
    import numpy as _np
    rng = _np.random.default_rng(0)
    yt = rng.random(60) * 10 + 5
    yp = yt + rng.standard_normal(60) * 0.3
    _p = _recipe("parity")
    with pytest.raises(ValueError) as e:
        _p.parity(yt, yp, band=("relatve", 0.10))
    assert "relatve" in str(e.value)


def test_parity_relative_and_absolute_bands_both_work():
    """不该触发的一侧：两个合法带型都要照常算，且结果不同。"""
    import numpy as _np
    rng = _np.random.default_rng(0)
    yt = rng.random(60) * 10 + 5
    yp = yt + rng.standard_normal(60) * 0.3
    _p = _recipe("parity")
    rel = _p.parity(yt, yp, band=("relative", 0.10))[2]["inside"]
    absv = _p.parity(yt, yp, band=("absolute", 0.10))[2]["inside"]
    assert rel != absv


def test_facet_metrics_rejects_an_unknown_scale():
    """`("指标", 值, "logarithmic")` 静默画成线性轴——用户要 log 轴是因为
    数据跨数量级，给成线性轴等于把图画错。
    """
    cr = _recipe("comparison_rank")
    with pytest.raises(ValueError) as e:
        cr.facet_metrics(["a", "b", "c"],
                         [("m", [1.0, 2.0, 3.0], "logarithmic")])
    assert "logarithmic" in str(e.value)


def test_facet_metrics_log_and_linear_still_work():
    cr = _recipe("comparison_rank")
    got = [cr.facet_metrics(["a", "b", "c"],
                            [("m", [1.0, 2.0, 3.0], s)])[1][0].get_xscale()
           for s in ("log", "linear")]
    assert got == ["log", "linear"]


def test_convergence_curves_rejects_an_unknown_mode():
    """`mode="minimum"` 抛的是裸 `KeyError: 'minimum'`，不列合法值——
    与 stat_box(loc=) 修前同一个毛病：能拦住，但用户只能去读源码。
    """
    import numpy as _np
    rng = _np.random.default_rng(1)
    ac = _recipe("algo_convergence")
    with pytest.raises(ValueError) as e:
        ac.convergence_curves([("A", list(rng.random(40) * 10), None)],
                              mode="minimum")
    assert "minimum" in str(e.value)
