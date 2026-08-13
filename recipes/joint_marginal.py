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

    # 共享轴不允许 aspect=equal（会挤散边缘直方图），改为强制 x/y
    # 等跨度，让分位圆接近正圆；轴限用 99.5% 分位数（抗离群点，
    # 极端落点不该决定整图比例）；密度读数走统计框，不放色条压数据
    xlo, xhi = np.quantile(x, [0.005, 0.995])
    ylo, yhi = np.quantile(y, [0.005, 0.995])
    xc, yc = (xlo + xhi) / 2, (ylo + yhi) / 2
    half = max(xhi - xlo, yhi - ylo) / 2 * 1.12
    # extent 对齐视窗：gridsize 作用于可见范围而非全数据范围，
    # 否则重尾离群点会把可见区的六边形撑得极粗
    hb = ax.hexbin(x, y, gridsize=gridsize, cmap=cmap_for("sequential2"),
                   mincnt=1, linewidths=0.1,
                   extent=(xc - half, xc + half, yc - half, yc + half))
    ax.set_xlim(xc - half, xc + half)
    ax.set_ylim(yc - half, yc + half)

    r = np.hypot(x, y)
    styles = ["--", ":"]
    lines = []
    stats = {}
    for q, ls in zip(quantiles, styles):
        rq = np.quantile(r, q)
        stats[q] = rq
        ax.add_patch(Circle((0, 0), rq, fill=False, color="0.25",
                            linestyle=ls, linewidth=0.9))
        lines.append(f"{q:.0%} 落点半径 = {rq:.2f}（{ls_name(ls)}圆）")
    if effective_r is not None:
        ax.add_patch(Circle((0, 0), effective_r, fill=False,
                            color=semantic("good"), linewidth=1.2))
        n_in = int(np.sum(r <= effective_r))
        stats["eff_frac"] = n_in / len(r)
        lines += [f"有效接收区 r ≤ {effective_r:g}",
                  f"落入 {n_in} / {len(r)} 条（{n_in/len(r):.1%}）"]
    lines.append(f"最远落点 {r.max():.1f}")
    stat_box(ax, lines, loc="lower right", fontsize=6.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # 边缘分布与主轴共享同一坐标（标准 jointplot 结构）
    bins = 60
    ax_top.hist(x, bins=bins, range=(xc - half, xc + half),
                color="#8FBFA8", edgecolor="white", linewidth=0.2)
    ax_right.hist(y, bins=bins, range=(yc - half, yc + half),
                  color="#8FBFA8", edgecolor="white",
                  linewidth=0.2, orientation="horizontal")
    fig._ff_stats = stats
    return fig, ax, stats


if __name__ == "__main__":
    apply_style()
    rng = np.random.default_rng(11)
    n = 40000
    # 各向异性高斯核 + t 分布重尾散射，模拟真实光斑（而非规整几何）
    core_pts = rng.normal(0, 1.0, (int(n * 0.75), 2)) * [1.35, 0.9]
    tail = rng.standard_t(df=4, size=(n - len(core_pts), 2)) * [2.4, 1.7]
    theta = np.deg2rad(18)                      # 光轴微倾，分布整体旋转
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    pts = np.vstack([core_pts, tail]) @ rot.T
    fig, ax, jstats = joint_hexbin(
        pts[:, 0], pts[:, 1],
        xlabel="接收面横坐标 η₁（m）",
        ylabel="接收面纵坐标 η₂（m）",
        effective_r=0.5)
    # 硬规则：图题中的数字必须来自计算变量
    fig.suptitle(f"落点向中心强汇聚：50% 落点半径 {jstats[0.5]:.1f} m，"
                 f"有效接收 {jstats['eff_frac']:.1%}",
                 fontsize=10, fontweight="bold", y=0.99)
    run_qa(fig, expect_width=("onehalf",))
    save_figure(fig, str(GALLERY / "joint_marginal"))
    print("joint_marginal: OK")
