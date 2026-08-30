"""路径规划图：多车/多回路空间路线 + 途经序号 + 里程统计（TSP/VRP/物流）。

论点合同示例：
- 结论：3 车分区配送，总里程与负载均衡度由数据算出。
  （具体数字由数据算出、经 f-string 进图题——写死在这里必然与图漂移，
   这正是 SPEC 2.1 要防的手写常数，只不过 docstring 逃过了 QA）
- 证据链：路线不交叉（分区合理）→ 序号可复现回路 → 统计框给分车里程。
替代：无序号无里程的散点连线（平庸，无法验证解）。
"""
from _common import GALLERY, PRESET
import numpy as np
from matplotlib.collections import LineCollection

from core import (text_color, ptx, ink, apply_style, new_figure, save_figure, run_qa, stat_box,
                  PALETTE, semantic)


def route_map(nodes, routes, depot=0, labels=None, xlabel="x（km）",
              ylabel="y（km）", width="onehalf", show_order=True):
    """nodes: (n,2) 坐标；routes: 若干条 [节点索引...]（含返回仓库则闭合）。

    depot: 仓库节点索引，画星标。返回 (fig, ax, info)：
    info["dists"] 各路线里程、info["total"] 总里程。
    """
    nodes = np.asarray(nodes, dtype=float)
    if not routes:
        raise ValueError("routes 为空——没有可画的路线")
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
        ax.plot(pts[:, 0], pts[:, 1], "o", color=c, markersize=ptx(3.5, "pt"),
                markeredgecolor="white", markeredgewidth=ptx(0.5, "lw"), zorder=3)
        if show_order:
            import matplotlib.patheffects as pe
            # 按"是否仓库"过滤：位置切片会漏掉不以仓库开头的路线首客户
            for i, idx in enumerate((k for k in r if k != depot), 1):
                ax.annotate(str(i), nodes[idx], xytext=(3, 3),
                            textcoords="offset points", fontsize=ptx(6.5),
                            color=text_color(c), zorder=6,
                            path_effects=[pe.withStroke(
                                linewidth=ptx(1.8, "lw"), foreground="white")])
    ax.plot(*nodes[depot], "*", color=semantic("highlight"), markersize=ptx(13, "pt"),
            markeredgecolor="white", markeredgewidth=ptx(0.7, "lw"), zorder=5)
    import matplotlib.patheffects as pe
    ax.annotate("仓库", nodes[depot], xytext=(8, -14),
                textcoords="offset points", fontsize=ptx(7), zorder=7,
                color=semantic("highlight"), fontweight="bold",
                path_effects=[pe.withStroke(linewidth=ptx(2.0, "lw"),
                                            foreground="white")])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # 空间图内寸土寸金：图例横排放到轴下方，不压任何路线
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=min(len(routes), 3), fontsize=ptx(6.5), frameon=False)
    d_max = max(dists)
    info = dict(dists=dists, total=float(sum(dists)),
                imbalance=float((d_max - min(dists)) / d_max)
                if d_max > 0 else 0.0)
    stat_box(ax, [f"{len(routes)} 条路线，总里程 {info['total']:.0f} km",
                  f"负载不均衡度 {info['imbalance']:.0%}"
                  "＝(最长−最短)/最长"],
             outside="top", fontsize=ptx(6.5))
    fig._ff_stats = info
    return fig, ax, info


if __name__ == "__main__":
    apply_style(PRESET)
    rng = np.random.default_rng(5)
    pts = np.vstack([[50, 50], rng.uniform(5, 95, (18, 2))])
    # 角度分区 + 扇区内按极角排序串联：回路天然不自交，
    # 与本图"路线不交叉（分区合理）"的证据链一致
    ang = np.arctan2(pts[1:, 1] - 50, pts[1:, 0] - 50)
    routes = []
    for lo, hi in [(-np.pi, -np.pi / 3), (-np.pi / 3, np.pi / 3),
                   (np.pi / 3, np.pi)]:
        sector = np.where((ang >= lo) & (ang < hi))[0]
        order = sector[np.argsort(ang[sector])] + 1
        routes.append([0] + list(order) + [0])
    fig, ax, info = route_map(pts, routes, depot=0)
    ax.set_title(f"3 车分区配送总里程 {info['total']:.0f} km，"
                 f"负载不均衡 {info['imbalance']:.0%}", fontsize=ptx(9.5))
    run_qa(fig, expect_width=("onehalf",))
    save_figure(fig, str(GALLERY / "route_map"))
    print("route_map: OK")
