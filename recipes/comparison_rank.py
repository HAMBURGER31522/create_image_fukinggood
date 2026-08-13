"""类别对比与排名：分组柱的期刊级替代。

包含 5 个 archetype：
- sorted_lollipop : 有序棒棒糖/点距（多类别单指标排名，替代竖柱）
- dumbbell        : 哑铃图（两条件对比，替代分组柱）
- slopegraph      : 斜率图（多类别两期排名/数值升降与交叉）
- butterfly       : 蝴蝶图（两类异质计数，零轴对开）
- facet_metrics   : 不可通约指标小倍数拆轴（硬规则：禁止共用 y）

论点合同示例（sorted_lollipop）：
- 结论：方案 C 的综合得分领先第二名 18%。
- 证据链：按值排序 → 条端直标 → 领先差距引线。
"""
from _common import GALLERY
import numpy as np
import matplotlib.pyplot as plt

from core import (apply_style, new_figure, save_figure, run_qa,
                  stat_box, callout, panel_label, PALETTE, semantic)


def sorted_lollipop(labels, values, unit="", highlight=None, title=""):
    """有序棒棒糖：按值降序、水平、条端直标。

    highlight: 强调项——None=自动强调最大值；int=排序前的原始索引；
    str=标签名。与 slopegraph 的 highlight（原始索引）语义一致。
    """
    if highlight is None:
        highlight = int(np.argmax(values))
    elif isinstance(highlight, str):
        highlight = list(labels).index(highlight)
    order = np.argsort(values)          # 水平图从下往上增大
    hi = int(np.where(order == highlight)[0][0])   # 排序后的位置
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]

    fig, ax = new_figure("single", ratio=0.7)
    y = np.arange(len(values))
    for i, (yi, v) in enumerate(zip(y, values)):
        c = semantic("highlight") if i == hi else PALETTE[0]
        ax.hlines(yi, 0, v, color=c, linewidth=1.6 if i == hi else 1.1,
                  alpha=1.0 if i == hi else 0.75)
        ax.plot([v], [yi], "o", color=c, markersize=5.5,
                markeredgecolor="white", markeredgewidth=0.8)
        ax.annotate(f"{v:g}{unit}", xy=(v, yi), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=7.5,
                    fontweight="bold" if i == hi else "normal", color=c)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.15)
    ax.set_title(title, fontsize=9.5)
    return fig, ax


def dumbbell(labels, before, after, cond_names=("前", "后"), unit="",
             xlabel="", higher_is_better=True):
    """哑铃图：每类别两点一线，直标 Δ。

    higher_is_better: 指标增大是否为"好"（决定 Δ 的语义配色）。
    """
    fig, ax = new_figure("single", ratio=0.7)
    y = np.arange(len(labels))
    c0, c1 = PALETTE[0], semantic("fit")
    for yi, b, a in zip(y, before, after):
        ax.plot([b, a], [yi, yi], color="0.78", linewidth=1.3, zorder=2)
        ax.plot([b], [yi], "o", color=c0, markersize=5.5, zorder=3,
                markeredgecolor="white", markeredgewidth=0.8)
        ax.plot([a], [yi], "o", color=c1, markersize=5.5, zorder=3,
                markeredgecolor="white", markeredgewidth=0.8)
        d = a - b
        good = (d >= 0) == higher_is_better
        ax.annotate(f"{'+' if d >= 0 else ''}{d:g}{unit}",
                    xy=(max(a, b), yi), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=7,
                    color=semantic("good") if good else semantic("bad"))
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.18)
    if xlabel:
        ax.set_xlabel(xlabel)
    # 线端语义代替图例；放左上空白，避免遮末行标注
    ax.plot([], [], "o", color=c0, label=cond_names[0])
    ax.plot([], [], "o", color=c1, label=cond_names[1])
    ax.legend(loc="upper left", ncols=2)
    return fig, ax


def slopegraph(labels, before, after, cond_names=("前", "后"), unit="",
               highlight=(), higher_is_better=True, width="single"):
    """斜率图：两期数值/排名连线，交叉即排名变化。替代两期分组柱。

    highlight: 需要强调的类别索引；其余灰化。
    """
    fig, ax = new_figure(width, ratio=0.85)
    for i, (lb, b, a) in enumerate(zip(labels, before, after)):
        emph = i in highlight
        good = (a >= b) == higher_is_better
        c = (semantic("good") if good else semantic("bad")) if emph else "0.7"
        ax.plot([0, 1], [b, a], "-o", color=c,
                linewidth=1.8 if emph else 1.0,
                markersize=5 if emph else 3.5,
                markeredgecolor="white", markeredgewidth=0.8,
                zorder=4 if emph else 2)
        ax.annotate(f"{lb} {b:g}{unit}", xy=(0, b), xytext=(-6, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=7.5 if emph else 7, color=c,
                    fontweight="bold" if emph else "normal")
        ax.annotate(f"{a:g}{unit}", xy=(1, a), xytext=(6, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=7.5 if emph else 7, color=c,
                    fontweight="bold" if emph else "normal")
    ax.set_xlim(-0.45, 1.3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(cond_names, fontsize=8.5)
    ax.grid(axis="x", visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])
    return fig, ax


def butterfly(labels, left, right, left_name, right_name, unit=""):
    """蝴蝶图：零轴对开的发散水平条，条端直标绝对值。"""
    fig, ax = new_figure("onehalf", ratio=0.62)
    y = np.arange(len(labels))
    cl, cr = "#4C9A82", "#7B6B9E"
    ax.barh(y, [-v for v in left], color=cl, height=0.6, alpha=0.85)
    ax.barh(y, right, color=cr, height=0.6, alpha=0.85)
    for yi, lv, rv in zip(y, left, right):
        ax.annotate(f"{lv:g}", xy=(-lv, yi), xytext=(-4, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=7, color=cl)
        ax.annotate(f"{rv:g}", xy=(rv, yi), xytext=(4, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=7, color=cr)
    ax.axvline(0, color="0.2", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.grid(axis="y", visible=False)
    # 左右翼按各自最大值定轴，避免小翼一侧大片空白
    ax.set_xlim(-max(max(left), 1) * 1.5, max(right) * 1.18)
    ax.set_xticks(ax.get_xticks())          # 先固定刻度再改标签（去符号）
    ax.set_xticklabels([f"{abs(t):g}" for t in ax.get_xticks()])
    ax.plot([], [], "s", color=cl, label=left_name)
    ax.plot([], [], "s", color=cr, label=right_name)
    ax.legend(loc="upper right", ncols=1)
    if unit:
        ax.set_xlabel(unit)
    return fig, ax


def facet_metrics(cat_labels, metrics, width="double"):
    """不可通约指标 → 小倍数拆轴。metrics: [(标题, 值列表, 'log'|'linear'), ...]

    每个指标一个面板，独立轴与单位；替代把 % / 元 / 无量纲塞进同一 y 轴。
    """
    n = len(metrics)
    from core import MM, COLUMN_WIDTHS
    w = COLUMN_WIDTHS[width] * MM
    fig, axes = plt.subplots(1, n, figsize=(w, w * 0.36))
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(wspace=0.35, top=0.85, bottom=0.15)
    y = np.arange(len(cat_labels))[::-1]
    for k, (ax, (title, vals, scale)) in enumerate(zip(axes, metrics)):
        if scale == "log":
            ax.set_xscale("log")
            title = f"{title}（log 轴）"
        ax.plot(vals, y, color="0.78", linewidth=1.3, zorder=2)
        for v, yi, c in zip(vals, y, PALETTE):
            ax.plot([v], [yi], "o", color=c, markersize=6.5, zorder=3,
                    markeredgecolor="white", markeredgewidth=1.0)
            ax.annotate(f"{v:g}", xy=(v, yi), xytext=(0, 8),
                        textcoords="offset points", ha="center",
                        fontsize=7.5, fontweight="bold", color=c)
        ax.set_yticks(y)
        ax.set_yticklabels(cat_labels if k == 0 else [""] * len(cat_labels))
        ax.set_title(title, fontsize=8.5)
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.22, y=0.3)
        panel_label(ax, chr(ord("a") + k), dx=-0.04 if k else -0.3)
    return fig, axes


if __name__ == "__main__":
    apply_style()
    rng = np.random.default_rng(0)

    vals = [72.1, 65.8, 88.4, 59.2, 74.9]
    top, second = sorted(vals)[-1], sorted(vals)[-2]
    lead, lead_pct = top - second, (top - second) / second * 100
    fig, ax = sorted_lollipop(
        ["方案A", "方案B", "方案C", "方案D", "方案E"], vals, unit=" 分",
        title=f"方案 C 综合得分领先第二名 {lead_pct:.0f}%")
    ax.set_xlabel("综合得分（分）")
    callout(ax, xy=(top, 4), text=f"领先 {lead:.1f} 分（+{lead_pct:.0f}%）",
            xytext=(0.55, 0.55),
            textcoords="axes fraction", color=semantic("highlight"))
    run_qa(fig, expect_width=("single",))
    save_figure(fig, str(GALLERY / "comparison_lollipop"))

    before, after = [3.2, 5.1, 4.4, 6.0], [2.1, 4.9, 2.8, 6.3]
    n_down = sum(a < b for a, b in zip(after, before))
    d_mean = np.mean(np.array(after) - np.array(before))
    fig, ax = dumbbell(
        ["城市A", "城市B", "城市C", "城市D"], before, after,
        cond_names=("优化前", "优化后"), unit=" h",
        xlabel="平均耗时（h）", higher_is_better=False)
    ax.set_title(f"优化后 {n_down}/{len(before)} 城市平均耗时下降",
                 fontsize=9.5)
    stat_box(ax, [f"n = {len(before)} 城市",
                  f"平均变化 {d_mean:+.2f} h"], loc="lower left",
             fontsize=6.5)
    run_qa(fig, expect_width=("single",))
    save_figure(fig, str(GALLERY / "comparison_dumbbell"))

    s_before = [72, 58, 66, 49, 61]
    s_after = [69, 71, 64, 52, 55]
    fig, ax = slopegraph(
        ["方案A", "方案B", "方案C", "方案D", "方案E"], s_before, s_after,
        cond_names=("政策前", "政策后"), unit=" 分",
        highlight=(1, 4))
    ax.set_title(f"政策后方案B 反超 A（{s_before[1]:g} → {s_after[1]:g} 分），"
                 f"方案E 下滑最多", fontsize=9)
    stat_box(ax, [f"n = {len(s_before)} 方案",
                  f"上升 {sum(a > b for a, b in zip(s_after, s_before))} 个 / "
                  f"下降 {sum(a < b for a, b in zip(s_after, s_before))} 个"],
             loc="lower left", fontsize=6.5)
    run_qa(fig, expect_width=("single",))
    save_figure(fig, str(GALLERY / "comparison_slopegraph"))

    d0s = np.arange(0, 0.57, 0.08)
    left_cnt = [270, 95, 0, 0, 0, 0, 0, 0]
    d0_last = d0s[max(i for i, v in enumerate(left_cnt) if v > 0)]
    fig, ax = butterfly(
        [f"D0 = {d:.2f} m" for d in d0s],
        left=left_cnt,
        right=[1770, 1580, 1430, 1250, 1135, 968, 625, 810],
        left_name="行程越界节点数", right_name="间距越界主索数",
        unit="数量（根）")
    ax.set_title(f"间距越界随 D0 增大总体下降，"
                 f"行程越界仅见于 D0 ≤ {d0_last:.2f} m", fontsize=9.5)
    ax.annotate(f"D0 ≥ {d0s[np.searchsorted(d0s, d0_last) + 1]:.2f} m "
                "后行程越界均为 0", xy=(0.98, 0.68),
                xycoords="axes fraction", ha="right", fontsize=7,
                color="#4C9A82", style="italic")
    stat_box(ax, [f"D0 扫描 {len(d0s)} 档（步长 {d0s[1]-d0s[0]:.2f} m）",
                  f"行程越界合计 {sum(left_cnt)} 节点"],
             loc="lower right", fontsize=6.5)
    run_qa(fig, expect_width=("onehalf",))
    save_figure(fig, str(GALLERY / "comparison_butterfly"))

    # 与 demo/beat_baseline 区分场景：算法对比（时间/内存/最优性 gap）
    t_solve = [312.0, 0.8]
    mem = [1850.0, 95.0]
    gaps = [0.0, 2.4]
    fig, axes = facet_metrics(
        ["分支定界（精确）", "贪心启发式"],
        [("求解时间（s）", t_solve, "log"),
         ("峰值内存（MB）", mem, "log"),
         ("最优性 gap（%）", gaps, "linear")])
    stat_box(axes[0], [f"提速 {t_solve[0]/t_solve[1]:.0f}×"],
             loc="center left", fontsize=6.5)
    fig.suptitle(f"贪心以 {gaps[1]:g}% gap 换 {t_solve[0]/t_solve[1]:.0f}× 提速"
                 f"与 {mem[0]/mem[1]:.0f}× 省存：大规模场景可用",
                 fontsize=10, fontweight="bold")
    run_qa(fig, expect_width=("double",))
    save_figure(fig, str(GALLERY / "comparison_facet_metrics"))
    print("comparison_rank: 5 figures OK")
