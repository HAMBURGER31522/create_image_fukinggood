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

from core import dot_interval, slope_lines, stat_box  # noqa: E402


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
    """几何扫描的**标准画法**就是 log 轴，而 log 刻度是 mathtext
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
    im = ax.imshow(np.linspace(0, 1, 16).reshape(4, 4), cmap="viridis")
    for i in range(4):
        ax.text(i, i, "0.9", color="white", ha="center", va="center")
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
