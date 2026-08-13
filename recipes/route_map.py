"""路径规划图：多车/多回路空间路线 + 途经序号 + 里程统计（TSP/VRP/物流）。

论点合同示例：
- 结论：3 车分区配送总里程 214 km，各车负载均衡（最大差 9%）。
- 证据链：路线不交叉（分区合理）→ 序号可复现回路 → 统计框给分车里程。
替代：无序号无里程的散点连线（平庸，无法验证解）。
"""
from _common import GALLERY
import numpy as np
from matplotlib.collections import LineCollection

from core import (apply_style, new_figure, save_figure, run_qa, stat_box,
                  PALETTE, semantic)


def route_map(nodes, routes, depot=0, labels=None, xlabel="x（km）",
              ylabel="y（km）", width="onehalf", show_order=True):
    """nodes: (n,2) 坐标；routes: 若干条 [节点索引...]（含返回仓库则闭合）。

    depot: 仓库节点索引，画星标。返回 (fig, ax, info)：
    info["dists"] 各路线里程、info["total"] 总里程。
    """
    nodes = np.asarray(nodes, dtype=float)
    fig, ax = new_figure(width, ratio=0.8)
    ax.set_aspect("equal")
    dists = []
    for k, r in enumerate(routes):
        r = np.asarray(r, dtype=int)
        pts = nodes[r]
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        d = float(np.sum(np.hypot(*(pts[1:] - pts[:-1]).T)))
        dists.append(d)
        c = PALETTE[k % len(PALETTE)]
        name = labels[k] if labels else f"路线{k + 1}"
        ax.add_collection(LineCollection(segs, colors=c, linewidths=1.2,
                                         alpha=0.85, zorder=2,
                                         label=f"{name}（{d:.0f} km）"))
        ax.plot(pts[:, 0], pts[:, 1], "o", color=c, markersize=3.5,
                markeredgecolor="white", markeredgewidth=0.5, zorder=3)
        if show_order:
            for i, idx in enumerate(r[1:-1] if r[0] == r[-1] else r[1:], 1):
                ax.annotate(str(i), nodes[idx], xytext=(3, 3),
                            textcoords="offset points", fontsize=6.5,
                            color=c, zorder=4)
    ax.plot(*nodes[depot], "*", color=semantic("highlight"), markersize=13,
            markeredgecolor="white", markeredgewidth=0.7, zorder=5)
    ax.annotate("仓库", nodes[depot], xytext=(6, -10),
                textcoords="offset points", fontsize=7,
                color=semantic("highlight"), fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # 空间图内寸土寸金：图例横排放到轴下方，不压任何路线
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=min(len(routes), 3), fontsize=6.5, frameon=False)
    info = dict(dists=dists, total=float(sum(dists)),
                imbalance=float((max(dists) - min(dists)) / max(dists)))
    stat_box(ax, [f"{len(routes)} 条路线，总里程 {info['total']:.0f} km",
                  f"负载不均衡度 {info['imbalance']:.0%}"
                  "＝(最长−最短)/最长"],
             loc="lower left", fontsize=6.5)
    fig._ff_stats = info
    return fig, ax, info


if __name__ == "__main__":
    apply_style()
    rng = np.random.default_rng(5)
    pts = np.vstack([[50, 50], rng.uniform(5, 95, (18, 2))])
    # 简单角度分区 + 最近邻串联，构造 3 条示例回路
    ang = np.arctan2(pts[1:, 1] - 50, pts[1:, 0] - 50)
    routes = []
    for k, (lo, hi) in enumerate([(-np.pi, -np.pi / 3),
                                  (-np.pi / 3, np.pi / 3),
                                  (np.pi / 3, np.pi)]):
        idx = np.where((ang >= lo) & (ang < hi))[0] + 1
        seq, rest = [0], list(idx)
        while rest:
            last = pts[seq[-1]]
            j = min(rest, key=lambda i: np.hypot(*(pts[i] - last)))
            seq.append(j)
            rest.remove(j)
        routes.append(seq + [0])
    fig, ax, info = route_map(pts, routes, depot=0)
    ax.set_title(f"3 车分区配送总里程 {info['total']:.0f} km，"
                 f"负载不均衡 {info['imbalance']:.0%}", fontsize=9.5)
    run_qa(fig, expect_width=("onehalf",))
    save_figure(fig, str(GALLERY / "route_map"))
    print("route_map: OK")
