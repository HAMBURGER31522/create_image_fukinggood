"""联合分布图：hexbin/散点 + 上缘直方图 + 分位半径圆 + 有效区计数框。

论点合同示例：
- 结论：调节后 50% 落点半径 0.97 m、90% 半径 5.19 m，有效接收 146346/412200。
- 证据链：hexbin 密度显示强中心汇聚 → 分位圆量化 → 计数框给出比例。
参考：skills/photo 图 25（焦面落点联合分布）。
替代：普通散点墨团（平庸）。
"""
from _common import GALLERY
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from core import (apply_style, save_figure, run_qa, marginal_grid,
                  stat_box, cmap_for, semantic)


def ls_name(ls):
    return {"--": "虚线", ":": "点线", "-": "实线"}.get(ls, ls)


def joint_hexbin(x, y, xlabel="x", ylabel="y", effective_r=None,
                 quantiles=(0.5, 0.9), gridsize=42, width="onehalf"):
    """中央 hexbin + 上/右边缘直方图（共享轴）+ 分位圆 + 计数框。"""
    fig, ax, ax_top, ax_right = marginal_grid(width, ratio=0.95, right=True)

    hb = ax.hexbin(x, y, gridsize=gridsize, cmap=cmap_for("sequential2"),
                   mincnt=1, linewidths=0.1)
    cax = fig.add_axes([0.13, 0.13, 0.016, 0.22])
    cb = fig.colorbar(hb, cax=cax)
    cb.set_label("计数", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)

    r = np.hypot(x, y)
    styles = ["--", ":"]
    lines = []
    for q, ls in zip(quantiles, styles):
        rq = np.quantile(r, q)
        ax.add_patch(Circle((0, 0), rq, fill=False, color="0.25",
                            linestyle=ls, linewidth=0.9))
        lines.append(f"{q:.0%} 落点半径 = {rq:.2f}（{ls_name(ls)}圆）")
    if effective_r is not None:
        ax.add_patch(Circle((0, 0), effective_r, fill=False,
                            color=semantic("good"), linewidth=1.2))
        n_in = int(np.sum(r <= effective_r))
        lines += [f"有效接收区 r ≤ {effective_r:g}",
                  f"落入 {n_in} / {len(r)} 条（{n_in/len(r):.1%}）"]
    lines.append(f"最远落点 {r.max():.1f}")
    stat_box(ax, lines, loc="lower right", fontsize=6.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # 边缘分布与主轴共享同一坐标（标准 jointplot 结构）
    bins = 60
    ax_top.hist(x, bins=bins, color="#8FBFA8", edgecolor="white",
                linewidth=0.2)
    ax_right.hist(y, bins=bins, color="#8FBFA8", edgecolor="white",
                  linewidth=0.2, orientation="horizontal")
    return fig, ax, ax_top


if __name__ == "__main__":
    apply_style()
    rng = np.random.default_rng(11)
    n = 40000
    core_pts = rng.normal(0, 1.1, (int(n * 0.7), 2))
    arms = []
    for ang in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        t = rng.gamma(2.2, 2.4, int(n * 0.3 / 8))
        w = rng.normal(0, 0.5, len(t))
        arms.append(np.c_[t * np.cos(ang) - w * np.sin(ang),
                          t * np.sin(ang) + w * np.cos(ang)])
    pts = np.vstack([core_pts] + arms)
    fig, ax, _ = joint_hexbin(
        pts[:, 0], pts[:, 1],
        xlabel="接收面横坐标 η₁（m）",
        ylabel="接收面纵坐标 η₂（m）",
        effective_r=0.5)
    fig.suptitle("落点向中心强汇聚：50% 落点半径 1.6 m，有效接收 6.9%",
                 fontsize=10, fontweight="bold", y=0.99)
    save_figure(fig, str(GALLERY / "joint_marginal"))
    run_qa(fig, expect_width=("onehalf",))
    print("joint_marginal: OK")
