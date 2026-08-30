"""分布对比：雨云图（半小提琴+箱+抖动点）与 ridgeline（多组漂移）。

论点合同示例（雨云）：
- 结论：组2 分布右移且方差最大，均值差异主要来自长尾而非整体平移。
- 证据链：半小提琴看形状 → 箱看分位 → 原始点看 n 与离群 → 统计框给均值±SD。
替代：箱线图藏样本（平庸）、均值柱+误差棒（更差）。
"""
from _common import GALLERY, PRESET
import numpy as np
from scipy import stats

from core import (ptx, apply_style, new_figure, save_figure, run_qa,
                  stat_box, PALETTE)


def raincloud(groups, labels, ylabel="值", width="onehalf"):
    """groups: 样本数组列表。横向排布：上半小提琴 + 中箱 + 下抖动点。

    组数超过色板长度时颜色循环；样本过少或方差为零的组退化为只画点+箱。
    """
    fig, ax = new_figure(width, ratio=0.62)
    rng = np.random.default_rng(0)
    for i, g in enumerate(groups):
        c = PALETTE[i % len(PALETTE)]
        g = np.asarray(g, dtype=float)
        g = g[np.isfinite(g)]
        if len(g) >= 3 and np.std(g) > 0:
            kde = stats.gaussian_kde(g)
            xs = np.linspace(np.min(g), np.max(g), 200)
            dens = kde(xs)
            dens = dens / dens.max() * 0.32
            ax.fill_between(xs, i + 0.08, i + 0.08 + dens, color=c,
                            alpha=0.55, lw=ptx(0, "lw"), zorder=3)
        bp = ax.boxplot(g, positions=[i], vert=False, widths=0.12,
                        showfliers=False, patch_artist=True, zorder=4,
                        boxprops=dict(facecolor="white", edgecolor=c,
                                      linewidth=ptx(1.0, "lw")),
                        whiskerprops=dict(color=c, linewidth=ptx(0.9, "lw")),
                        capprops=dict(color=c, linewidth=ptx(0.9, "lw")),
                        medianprops=dict(color=c, linewidth=ptx(1.4, "lw")))
        jitter = rng.uniform(-0.05, 0.05, len(g)) - 0.22
        ax.plot(g, i + jitter, "o", color=c, markersize=2.2, alpha=0.45,
                markeredgewidth=ptx(0, "lw"), zorder=2)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.55, len(groups) - 0.2)
    ax.set_xlabel(ylabel)
    ax.grid(axis="y", visible=False)
    stat_box(ax, [f"{lb}: n={len(g)}, 均值 {np.mean(g):.2f} ± {np.std(g):.2f}"
                  for lb, g in zip(labels, groups)],
             loc="lower right", fontsize=ptx(6.5))
    return fig, ax


def ridgeline(groups, labels, xlabel="值", width="single", cmap_colors=None):
    """岭线图：>6 组分布随序漂移。组序按时间/条件排列。"""
    fig, ax = new_figure(width, ratio=0.9)
    colors = cmap_colors or [PALETTE[i % len(PALETTE)] for i in range(len(groups))]
    lo = min(np.min(g) for g in groups)
    hi = max(np.max(g) for g in groups)
    xs = np.linspace(lo, hi, 300)
    for i, (g, c) in enumerate(zip(groups, colors)):
        dens = stats.gaussian_kde(g)(xs)
        dens = dens / dens.max() * 1.25
        base = len(groups) - 1 - i
        ax.fill_between(xs, base, base + dens, color=c, alpha=0.75, lw=ptx(0, "lw"),
                        zorder=len(groups) - i)
        ax.plot(xs, base + dens, color="white", linewidth=ptx(0.7, "lw"),
                zorder=len(groups) - i)
        med = np.median(g)
        ax.plot([med], [base], "|", color="0.2", markersize=8, zorder=99)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(labels[::-1])
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    fig.text(0.99, 0.005, "黑色短竖线 = 各组中位数", ha="right",
             fontsize=ptx(6.5), color="0.35")
    return fig, ax


if __name__ == "__main__":
    apply_style(PRESET)
    rng = np.random.default_rng(5)
    groups = [rng.normal(3.0, 0.5, 160),
              np.concatenate([rng.normal(3.6, 0.6, 130), rng.normal(5.4, 0.4, 40)]),
              rng.normal(3.3, 0.35, 150)]
    fig, ax = raincloud(groups, ["组1", "组2", "组3"], ylabel="轴长（µm）")
    ax.set_title("组2 均值差异来自长尾成分而非整体平移", fontsize=ptx(9.5))
    run_qa(fig, expect_width=("onehalf",))
    save_figure(fig, str(GALLERY / "raincloud"))

    groups = [rng.normal(3 + 0.25 * k, 0.4 + 0.03 * k, 200) for k in range(8)]
    from core import cmap_for
    cm = cmap_for("sequential2")
    colors = [cm(v) for v in np.linspace(0.15, 0.85, 8)]
    fig, ax = ridgeline(groups, [f"t = {k}" for k in range(8)],
                        xlabel="残差（cm）", cmap_colors=colors)
    ax.set_title("分布随迭代整体右移且方差增大", fontsize=ptx(9.5))
    meds = [np.median(g) for g in groups]
    stat_box(ax, [f"每组 n = {len(groups[0])}",
                  f"中位数漂移 {meds[-1]-meds[0]:+.1f} cm（t0 → t7）"],
             loc="lower right", fontsize=ptx(6.5))
    run_qa(fig, expect_width=("single",))
    save_figure(fig, str(GALLERY / "ridgeline"))
    print("raincloud: 2 figures OK")
