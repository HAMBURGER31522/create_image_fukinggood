"""空间网络小倍数：真实坐标、贯穿簇高亮、统一轴限、角标统计。

论点合同示例：
- 结论：组2 率先形成贯穿簇（红），组1/组3 仅局部连通；三组同轴限可直接比密度。
- 证据链：边用 LineCollection 灰色底层 → 贯穿簇亮色高 zorder → 角标给边数/占比。
对照：A 题「三组介质网络拓扑」（有论点但轴限不一致、缺统计框）。
"""
from _common import GALLERY
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from core import (apply_style, save_figure, run_qa, small_multiples,
                  semantic)


def spatial_networks(nets, titles, xlim, ylim, ncols=3, width="double",
                     boundary_frac=0.04):
    """nets: [(nodes(n,2), edges(m,2 索引), spanning_mask(m,)), ...]

    spanning_mask 标记属于贯穿簇的边。所有面板共用 xlim/ylim（硬规则）。
    """
    fig, axes = small_multiples(len(nets), ncols=ncols, width=width,
                                ratio=1.0)
    for ax, (nodes, edges, span_mask), t in zip(axes, nets, titles):
        segs = nodes[edges]
        other = LineCollection(segs[~span_mask], colors="0.72",
                               linewidths=0.5, zorder=2)
        ax.add_collection(other)
        if span_mask.any():
            span = LineCollection(segs[span_mask], colors=semantic("bad"),
                                  linewidths=0.9, zorder=3)
            ax.add_collection(span)
        ax.plot(nodes[:, 0], nodes[:, 1], "o", color="0.45", markersize=1.2,
                markeredgewidth=0, zorder=4)
        # 两侧电极带
        wband = (xlim[1] - xlim[0]) * boundary_frac
        ax.axvspan(xlim[0], xlim[0] + wband, color="#E8D8A0", alpha=0.5,
                   zorder=1)
        ax.axvspan(xlim[1] - wband, xlim[1], color="#E8D8A0", alpha=0.5,
                   zorder=1)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_title(t, fontsize=8)
        n_span = int(span_mask.sum())
        ax.text(0.03, 0.03,
                f"边数：{len(edges)}\n贯穿簇边：{n_span}"
                f"（{n_span/max(len(edges),1):.0%}）",
                transform=ax.transAxes, fontsize=6.5, va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="0.75", alpha=0.85, linewidth=0.5))
    fig.text(0.5, 0.015, "注：两侧米黄色竖带为电极接触带；红色边属于贯穿簇",
             ha="center", fontsize=7, color="0.35")
    return fig, axes


def _demo_net(rng, n, radius, span=False):
    nodes = rng.uniform(0, 100, (n, 2))
    from scipy.spatial import cKDTree
    tree = cKDTree(nodes)
    pairs = np.array(sorted(tree.query_pairs(radius)))
    if len(pairs) == 0:
        pairs = np.zeros((0, 2), dtype=int)
    mask = np.zeros(len(pairs), dtype=bool)
    if span and len(pairs):
        mid = nodes[pairs].mean(axis=1)
        mask = np.abs(mid[:, 1] - 50) < 18
    return nodes, pairs, mask


if __name__ == "__main__":
    apply_style()
    rng = np.random.default_rng(6)
    nets = [_demo_net(rng, 260, 8.0),
            _demo_net(rng, 420, 8.5, span=True),
            _demo_net(rng, 300, 7.5)]
    fig, _ = spatial_networks(
        nets, ["组1（φ = 0.55%）", "组2（φ = 0.83%）", "组3（φ = 0.62%）"],
        xlim=(0, 100), ylim=(0, 100))
    fig.suptitle("仅组2 形成贯穿簇（红）：三组同轴限下密度差异直接可比",
                 fontsize=10, fontweight="bold")
    fig.subplots_adjust(top=0.82, bottom=0.1)
    run_qa(fig, expect_width=("double",))
    save_figure(fig, str(GALLERY / "network_percolation"))
    print("network_percolation: OK")
