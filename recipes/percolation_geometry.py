"""渗流几何：三维线段介质体 + 贯通簇高亮 + 边缘分布。

**画研究对象本身，而不是它的汇总数。** 渗流问题的核心图是贯通簇的空间
形态——哪几根连成了一片、从哪一侧咬到哪一侧。把它压成"最大簇 258 根"
这样一个标量再画成棒棒糖，是信息密度塌方的主因：同样一栏宽，前者承载
数百根线段的空间结构，后者只承载三个数。

对标参考图：三维面片抛物面、三维光路场景——满屏都是研究对象本身。

可直接运行看 demo：
    python recipes/percolation_geometry.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import (apply_style, save_figure, run_qa, stat_box, panel_label,
                  semantic, emphasis, MM, COLUMN_WIDTHS)


def segment_clusters(seg, gap, radius: float = 0.0):
    """按"表面间距 ≤ gap"把线段并成连通簇（胶囊体近似）。

    seg: (n, 2, 3) 线段端点（圆柱**轴线**）；gap: 表面间距判据；
    radius: 圆柱半径，缺省 0（把线段当零粗细）。

    **radius 必须传对。** 题面给的判据几乎总是"表面间距"，而线段距离
    算的是轴线距离，两者差 2×radius。漏掉半径会让连通性判定整体崩掉：
    实测半径 30 nm、判据 1.8 nm 的算例，漏传 radius 后 535 根碎成
    531 个簇（正确结果是最大簇 258 根）——图还画得出来，结论全错。
    """
    thresh = gap + 2.0 * radius
    n = len(seg)
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # 先按包围盒粗筛，再算精确线段距离——n 上千时全对全是 O(n²) 但可接受
    lo = seg.min(axis=1) - thresh
    hi = seg.max(axis=1) + thresh
    for i in range(n):
        cand = np.where((lo[:, 0] <= hi[i, 0]) & (hi[:, 0] >= lo[i, 0]) &
                        (lo[:, 1] <= hi[i, 1]) & (hi[:, 1] >= lo[i, 1]) &
                        (lo[:, 2] <= hi[i, 2]) & (hi[:, 2] >= lo[i, 2]))[0]
        for j in cand[cand > i]:
            if _seg_dist(seg[i], seg[j]) <= thresh:
                union(i, j)
    return np.array([find(i) for i in range(n)])


def _seg_dist(p, q):
    """两条线段间最短距离（含平行退化）。"""
    u, v = p[1] - p[0], q[1] - q[0]
    w = p[0] - q[0]
    a, b, c = u @ u, u @ v, v @ v
    d, e = u @ w, v @ w
    den = a * c - b * b
    if den < 1e-12:
        s, t = 0.0, (e / c if c > 1e-12 else 0.0)
    else:
        s = np.clip((b * e - c * d) / den, 0, 1)
        t = np.clip((a * e - b * d) / den, 0, 1)
    return float(np.linalg.norm(w + s * u - t * v))


def spanning_cluster_3d(seg, gap, radius: float = 0.0, axis=0, span_lim=None,
                        width="onehalf", elev=20, azim=-62, unit="nm",
                        title=None):
    """三维介质体 + 贯通簇高亮 + 右侧簇尺寸边缘分布。

    seg: (n, 2, 3) 线段端点；gap: 表面间距判据；radius: 圆柱半径；
    axis: 贯通方向（0=x）；span_lim: (lo, hi) 两侧电极面坐标，缺省取数据极值。

    返回 (fig, ax3d, ax_marg, info)。info 含最大簇根数、是否贯通等。
    """
    lab = segment_clusters(seg, gap, radius)
    lo, hi = (span_lim if span_lim is not None
              else (seg[:, :, axis].min(), seg[:, :, axis].max()))
    tol = (hi - lo) * 1e-6
    touch_lo = (np.abs(seg[:, :, axis] - lo) <= tol).any(axis=1)
    touch_hi = (np.abs(seg[:, :, axis] - hi) <= tol).any(axis=1)

    sizes = {}
    span_id = None
    for c in np.unique(lab):
        m = lab == c
        sizes[c] = int(m.sum())
        if touch_lo[m].any() and touch_hi[m].any():
            if span_id is None or sizes[c] > sizes[span_id]:
                span_id = c
    big_id = max(sizes, key=sizes.get)
    hot = lab == (span_id if span_id is not None else big_id)

    w = COLUMN_WIDTHS.get(width, width) * MM
    fig = plt.figure(figsize=(w, w * 0.58))
    ax = fig.add_axes([0.01, 0.06, 0.70, 0.86], projection="3d")
    axm = fig.add_axes([0.76, 0.16, 0.21, 0.66])

    c_hot = semantic("highlight") if span_id is None else semantic("good")
    ax.add_collection3d(Line3DCollection(
        seg[~hot], colors=emphasis("#9E9E9E", "background"), linewidths=0.35))
    ax.add_collection3d(Line3DCollection(
        seg[hot], colors=c_hot, linewidths=1.2))
    # 两侧电极面：半透明平面，让"贯通"这件事在图上有实体依据
    other = [k for k in range(3) if k != axis]
    g1 = np.linspace(seg[:, :, other[0]].min(), seg[:, :, other[0]].max(), 2)
    g2 = np.linspace(seg[:, :, other[1]].min(), seg[:, :, other[1]].max(), 2)
    G1, G2 = np.meshgrid(g1, g2)
    for pv in (lo, hi):
        coords = [None, None, None]
        coords[axis] = np.full_like(G1, pv)
        coords[other[0]], coords[other[1]] = G1, G2
        ax.plot_surface(*coords, color="#E8C35A", alpha=0.16, shade=False,
                        linewidth=0)
    ax.set_xlim(seg[:, :, 0].min(), seg[:, :, 0].max())
    ax.set_ylim(seg[:, :, 1].min(), seg[:, :, 1].max())
    ax.set_zlim(seg[:, :, 2].min(), seg[:, :, 2].max())
    span = [np.ptp(seg[:, :, k]) for k in range(3)]
    ax.set_box_aspect([s / max(span) * 2 + 0.6 for s in span])
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel(f"X / {unit}", fontsize=6.5, labelpad=-2)
    ax.set_ylabel(f"Y / {unit}", fontsize=6.5, labelpad=-3)
    ax.tick_params(labelsize=6.5, pad=-2)
    ax.set_zticklabels([])
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0.03)

    # 边缘分布：簇尺寸谱——一眼看出是"一枝独大"还是"碎成一片"
    vals = np.array(sorted(sizes.values()))[::-1]
    axm.barh(np.arange(len(vals)), vals, height=0.85,
             color=[c_hot if v == sizes[span_id if span_id is not None
                                        else big_id] else
                    emphasis("#9E9E9E", "context") for v in vals])
    axm.set_xscale("log")
    axm.invert_yaxis()
    axm.set_yticks([])
    axm.set_xlabel("簇内根数（对数轴）", fontsize=6.5)
    axm.set_title(f"簇尺寸谱（{len(vals)} 簇）", fontsize=7)
    axm.tick_params(labelsize=6.5)
    axm.spines[["top", "right"]].set_visible(False)

    info = dict(n=len(seg), n_cluster=len(sizes), biggest=sizes[big_id],
                spanning=span_id is not None,
                span_size=int(hot.sum()), gap=gap, radius=radius,
                touch_lo=int(touch_lo.sum()), touch_hi=int(touch_hi.sum()))
    stat_box(ax, [f"介质 {info['n']} 根，表面间距判据 {gap:g} {unit}"
                  f"（半径 {radius:g} {unit}）",
                  f"簇数 {info['n_cluster']}，最大簇 {info['biggest']} 根",
                  f"触左 {info['touch_lo']} / 触右 {info['touch_hi']} 根",
                  ("判定：贯通（高亮簇同时咬住两侧电极面）"
                   if info["spanning"] else "判定：不贯通")],
             outside="top", fontsize=6.5)
    if title:
        fig.suptitle(title, y=0.99, fontsize=9.5)
    fig._ff_stats = info
    return fig, ax, axm, info


if __name__ == "__main__":
    from _common import GALLERY

    apply_style("cn")
    rng = np.random.default_rng(7)
    L, n = 5000.0, 320
    ctr = rng.uniform(-L, L, size=(n, 3)) * [1.0, 0.55, 0.55]
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    half = rng.uniform(600, 1400, size=(n, 1))
    seg = np.stack([ctr - d * half, ctr + d * half], axis=1)
    seg[:, :, 0] = np.clip(seg[:, :, 0], -L, L)

    fig, ax, axm, info = spanning_cluster_3d(
        seg, gap=320.0, radius=0.0, span_lim=(-L, L), width="onehalf",
        title=None)
    fig.suptitle(f"{info['n']} 根介质结成 {info['n_cluster']} 簇，"
                 f"最大簇 {info['biggest']} 根"
                 f"{'并贯通两侧电极面' if info['spanning'] else '未贯通'}",
                 y=0.99, fontsize=9.5)
    run_qa(fig, expect_width=("onehalf",))
    save_figure(fig, str(GALLERY / "percolation_geometry"))
    print("percolation_geometry: 1 figure OK")
