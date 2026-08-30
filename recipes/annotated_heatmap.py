"""注释热力矩阵：混淆矩阵 与 下三角相关矩阵（seaborn 默认热力的期刊级替代）。

archetype 1: confusion_matrix —— 行归一色阶 + 计数/行占比双标注 + 对角强调
archetype 2: corr_matrix      —— 下三角掩膜 + 发散色 + 数值直标 + 强相关强调

论点合同示例（confusion）：
- 结论：总体准确率，主要混淆发生在最大混淆对。
  （具体数字由数据算出、经 f-string 进图题——写死在这里必然与图漂移，
   这正是 SPEC 2.1 要防的手写常数，只不过 docstring 逃过了 QA）
- 证据链：对角块深色 → 非对角唯一深块引导视线 → 统计框给宏平均。
"""
from _common import GALLERY
import numpy as np
from matplotlib.patches import Rectangle

from core import (apply_style, new_figure, save_figure, run_qa,
                  stat_box, cmap_for, truncate_cmap, semantic)


def confusion_matrix(M, class_names, xlabel="预测类别", ylabel="真实类别",
                     width="single"):
    """M[i][j] = 真实 i 被判为 j 的计数。行归一上色，格内双标注。"""
    M = np.asarray(M, dtype=float)
    share = M / M.sum(axis=1, keepdims=True)
    n = len(class_names)
    fig, ax = new_figure(width, ratio=0.9)
    im = ax.imshow(share, cmap=truncate_cmap(cmap_for("sequential2"), 0.0, 0.85),
                   vmin=0, vmax=1)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("行占比", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    for i in range(n):
        for j in range(n):
            dark = share[i, j] > 0.45
            ax.text(j, i, f"{M[i, j]:g}\n{share[i, j]:.0%}",
                    ha="center", va="center", fontsize=7.5, linespacing=1.4,
                    color="white" if dark else "0.25",
                    fontweight="bold" if i == j else "normal")
        # 对角强调框
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                               edgecolor="0.2", linewidth=1.2))
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    acc = np.trace(M) / M.sum()
    recalls = np.diag(M) / M.sum(axis=1)
    # 统计放图外底部：格子全被数据占用，图内任何框都会压住单元格
    fig.text(0.02, 0.02,
             f"n = {int(M.sum())} 样本 · 总体准确率 {acc:.0%} · "
             f"宏平均召回 {recalls.mean():.0%}",
             fontsize=6.5, ha="left", va="bottom", color="0.3",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="0.75", alpha=0.85, linewidth=0.5))
    fig.subplots_adjust(bottom=0.24)
    info = dict(acc=acc, macro_recall=recalls.mean(), share=share,
                n_samples=int(M.sum()))
    fig._ff_stats = info
    return fig, ax, info


def corr_matrix(R, names, width="single", emph_thresh=0.7):
    """下三角相关矩阵：掩膜上三角，|r| ≥ emph_thresh 加强调框。"""
    R = np.asarray(R, dtype=float)
    n = len(names)
    mask = np.triu(np.ones_like(R, dtype=bool), k=1)   # 保留对角，首行不空
    Rm = np.ma.masked_where(mask, R)
    fig, ax = new_figure(width, ratio=0.9)
    im = ax.imshow(Rm, cmap=cmap_for("diverging"), vmin=-1, vmax=1)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Pearson r", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    n_strong = 0
    for i in range(n):
        ax.text(i, i, "1", ha="center", va="center", fontsize=6.5,
                color="white")
        for j in range(i):
            strong = abs(R[i, j]) >= emph_thresh
            n_strong += strong
            ax.text(j, i, f"{R[i, j]:+.2f}", ha="center", va="center",
                    fontsize=7, color="0.15",
                    fontweight="bold" if strong else "normal")
            if strong:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor=semantic("highlight"),
                                       linewidth=1.2))
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_yticklabels(names)
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # 格内数字直接来自 R 本身，把矩阵纳入来源（否则每个格值都会
    # 被判成手写常数）
    info = dict(n_strong=int(n_strong), thresh=emph_thresh, R=R)
    fig._ff_stats = info
    return fig, ax, info


if __name__ == "__main__":
    apply_style()
    M = [[112, 6, 2], [9, 87, 12], [3, 5, 96]]
    names = ["类1", "类2", "类3"]
    fig, ax, info = confusion_matrix(M, names)
    worst = np.unravel_index(
        np.argmax(info["share"] - 2 * np.eye(len(names))), info["share"].shape)
    ax.set_title(f"总体准确率 {info['acc']:.0%}，"
                 f"主要混淆为{names[worst[0]]}→{names[worst[1]]}"
                 f"（{info['share'][worst]:.0%}）", fontsize=9)
    run_qa(fig, expect_width=("single",))
    save_figure(fig, str(GALLERY / "confusion_matrix"))

    rng = np.random.default_rng(12)
    A = rng.normal(0, 1, (200, 5))
    A[:, 1] += 0.9 * A[:, 0]
    A[:, 4] -= 0.8 * A[:, 3]
    R = np.corrcoef(A.T)
    vnames = ["降雨量", "径流量", "坡度", "植被覆盖", "侵蚀量"]
    fig, ax, info = corr_matrix(R, vnames)
    info["n_obs"] = len(A)
    fig._ff_stats = info
    ax.set_title(f"{info['n_strong']} 对强相关（|r| ≥ 0.7）：需在回归前处理共线性",
                 fontsize=9)
    stat_box(ax, [f"n = {len(A)} 观测", "Pearson r，下三角"],
             loc="upper right", fontsize=6.5)
    run_qa(fig, expect_width=("single",))
    save_figure(fig, str(GALLERY / "corr_matrix"))
    print("annotated_heatmap: 2 figures OK")
