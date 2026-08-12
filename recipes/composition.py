"""组成与占比：饼图的期刊级替代 + 平行坐标（雷达替代）。

archetype 1: share_bars       —— 有序水平条 + 直标 %与n（占比默认替代）
archetype 2: waffle           —— 华夫图 10×10（部分-总体印象，≤4 类）
archetype 3: stacked_share    —— 堆叠水平条（跨组构成对比）
archetype 4: parallel_coords  —— 平行坐标（3+ 指标轮廓，替代雷达）

硬拒绝：饼图（>3 类或需精确比较）、3D 饼、雷达图做排名。
"""
from _common import GALLERY
import numpy as np
import matplotlib.pyplot as plt

from core import (apply_style, new_figure, save_figure, run_qa,
                  stat_box, PALETTE, OKABE_ITO, semantic)


def share_bars(labels, counts, unit="", width="single", highlight=None):
    """有序水平条：占比 + 绝对数直标。饼图的默认替代。"""
    total = sum(counts)
    order = np.argsort(counts)
    labels = [labels[i] for i in order]
    counts = [counts[i] for i in order]
    fig, ax = new_figure(width, ratio=0.62)
    y = np.arange(len(labels))
    colors = [semantic("highlight") if labels[i] == highlight else PALETTE[0]
              for i in range(len(labels))]
    ax.barh(y, counts, color=colors, height=0.6, alpha=0.9)
    for yi, v in zip(y, counts):
        ax.annotate(f"{v/total:.1%}（{v:g}{unit}）", xy=(v, yi),
                    xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=7.2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.25)
    ax.set_xlabel(f"数量{('（' + unit + '）') if unit else ''}")
    return fig, ax


def waffle(labels, counts, width="single", n=10):
    """华夫图：n×n 格，每格 = 总量/n²。仅 ≤4 类。"""
    total = sum(counts)
    cells = np.round(np.asarray(counts) / total * n * n).astype(int)
    cells[-1] = n * n - cells[:-1].sum()
    grid = np.repeat(np.arange(len(counts)), cells)[: n * n].reshape(n, n)
    colors = [OKABE_ITO[k] for k in ["blue", "orange", "green", "purple"]]
    fig, ax = new_figure(width, ratio=0.8)
    for i in range(n):
        for j in range(n):
            ax.add_patch(plt.Rectangle((j, n - 1 - i), 0.9, 0.9,
                                       facecolor=colors[grid[i, j]],
                                       edgecolor="white", linewidth=1.2))
    ax.set_xlim(-0.2, n + 0.1)
    ax.set_ylim(-0.2, n + 0.1)
    ax.set_aspect("equal")
    ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[k])
               for k in range(len(labels))]
    ax.legend(handles,
              [f"{lb} {c/total:.0%}" for lb, c in zip(labels, counts)],
              loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7.5,
              frameon=False)
    return fig, ax


def stacked_share(group_labels, cat_labels, matrix, width="onehalf"):
    """堆叠水平条：多组构成对比。matrix[g][c] = 计数。"""
    matrix = np.asarray(matrix, dtype=float)
    shares = matrix / matrix.sum(axis=1, keepdims=True)
    fig, ax = new_figure(width, ratio=0.5)
    y = np.arange(len(group_labels))
    left = np.zeros(len(group_labels))
    for c in range(len(cat_labels)):
        ax.barh(y, shares[:, c], left=left, height=0.55,
                color=PALETTE[c % len(PALETTE)], label=cat_labels[c])
        for yi, s, l in zip(y, shares[:, c], left):
            if s > 0.07:
                ax.text(l + s / 2, yi, f"{s:.0%}", ha="center", va="center",
                        fontsize=6.8, color="white", fontweight="bold")
        left += shares[:, c]
    ax.set_yticks(y)
    ax.set_yticklabels(group_labels)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncols=len(cat_labels), fontsize=7)
    return fig, ax


def parallel_coords(names, data, dims, highlight_idx=(), width="onehalf"):
    """平行坐标：data[i][d] 原值；每维独立归一。highlight_idx 强调方案。"""
    data = np.asarray(data, dtype=float)
    norm = (data - data.min(axis=0)) / (np.ptp(data, axis=0) + 1e-12)
    fig, ax = new_figure(width, ratio=0.55)
    x = np.arange(len(dims))
    for i in range(len(data)):
        if i in highlight_idx:
            continue
        ax.plot(x, norm[i], color="#CCCCCC", linewidth=0.8, alpha=0.6,
                zorder=2)
    for j, i in enumerate(highlight_idx):
        ax.plot(x, norm[i], color=PALETTE[j % len(PALETTE)], linewidth=1.8,
                zorder=3, label=names[i])
    for xi, d in zip(x, dims):
        ax.axvline(xi, color="0.55", linewidth=0.6)
        ax.text(xi, 1.04, f"{data[:, xi].max():.3g}", ha="center",
                fontsize=6, color="0.4")
        ax.text(xi, -0.08, f"{data[:, xi].min():.3g}", ha="center",
                fontsize=6, color="0.4")
    ax.set_xticks(x)
    ax.set_xticklabels(dims, fontsize=7.5)
    ax.set_yticks([])
    ax.grid(False)
    ax.legend(loc="upper right", fontsize=7)
    ax.set_ylim(-0.14, 1.14)
    return fig, ax


if __name__ == "__main__":
    apply_style()
    fig, ax = share_bars(
        ["短段贴边界面", "内部完整段", "跨界截断段", "孤立段"],
        [1240, 3105, 462, 89], unit=" 根", highlight="短段贴边界面")
    ax.set_title("内部完整段占 63%；贴边界面短段占 25% 为截断证据",
                 fontsize=9)
    save_figure(fig, str(GALLERY / "composition_share_bars"))
    run_qa(fig, expect_width=("single",))

    fig, ax = waffle(["介质A", "介质B", "基体"], [12, 27, 61])
    ax.set_title("成本构成：基体占六成，介质B 为主要增量", fontsize=9)
    save_figure(fig, str(GALLERY / "composition_waffle"))
    run_qa(fig, expect_width=("single",))

    fig, ax = stacked_share(
        ["组1", "组2", "组3"], ["孤立", "小簇（2–10）", "大簇（>10）"],
        [[62, 30, 8], [33, 37, 30], [55, 33, 12]])
    ax.set_title("组2 大簇占比三倍于其余组", fontsize=9)
    save_figure(fig, str(GALLERY / "composition_stacked"))
    run_qa(fig, expect_width=("onehalf",))

    rng = np.random.default_rng(8)
    data = rng.uniform(0, 1, (24, 5)) * [10, 5, 100, 40, 1]
    data[5] = [2.1, 4.6, 88, 12, 0.93]
    fig, ax = parallel_coords(
        [f"方案{i}" for i in range(24)], data,
        ["成本", "时间", "覆盖率", "风险", "稳健性"], highlight_idx=(5,))
    ax.set_title("方案5 以低成本高覆盖进入 Pareto 集", fontsize=9)
    save_figure(fig, str(GALLERY / "parallel_coords"))
    run_qa(fig, expect_width=("onehalf",))
    print("composition: 4 figures OK")
