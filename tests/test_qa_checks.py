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
