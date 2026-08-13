"""聚类结果图：簇着色散点 + 簇心星标 + 2σ 协方差椭圆 + 轮廓系数统计框。

论点合同示例：
- 结论：K=3 聚类结构清晰（轮廓系数 0.61），簇间无重叠。
- 证据链：着色散点显示分离 → 协方差椭圆量化簇形状 → 轮廓系数给全局质量。
替代：默认 tab10 散点无簇心无质量指标（平庸，无法评价聚类好坏）。
"""
from _common import GALLERY
import numpy as np
from matplotlib.patches import Ellipse

from core import (apply_style, new_figure, save_figure, run_qa, stat_box,
                  PALETTE, semantic)


def _silhouette(X, labels):
    """numpy 版轮廓系数均值（n ≲ 数千可用，与 sklearn 定义一致）。

    噪声点（-1）不参与；单点簇 sᵢ=0（标准定义）；簇数 < 2 或
    全噪声时轮廓系数无定义，返回 NaN。
    """
    m = labels >= 0
    X, labels = X[m], labels[m]
    ks = np.unique(labels)
    if len(X) == 0 or len(ks) < 2:
        return float("nan")
    d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    s = np.zeros(len(X))
    for i in range(len(X)):
        same = labels == labels[i]
        same[i] = False
        if not same.any():          # 单点簇：s_i = 0
            s[i] = 0.0
            continue
        a = d[i, same].mean()
        b = min(d[i, labels == k].mean() for k in ks if k != labels[i])
        denom = max(a, b)
        s[i] = (b - a) / denom if denom > 0 else 0.0
    return float(s.mean())


def _cov_ellipse(ax, pts, color, n_std=2.0):
    """按样本协方差画 n_std 协方差椭圆（刻画簇形状与朝向，
    非均值置信区间；二维正态下 2σ 覆盖约 86%，不是 95%）。"""
    if len(pts) < 3:
        return
    cov = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
    w, h = 2 * n_std * np.sqrt(np.maximum(vals[::-1], 0))
    ax.add_patch(Ellipse(pts.mean(axis=0), w, h, angle=ang, fill=False,
                         color=color, linewidth=1.0, linestyle="--",
                         alpha=0.8))


def cluster_scatter(X, labels, xlabel="特征 1", ylabel="特征 2",
                    cluster_names=None, width="onehalf", n_std=2.0):
    """X: (n,2)；labels: 簇编号（-1 为噪声，画灰点不入统计）。

    返回 (fig, ax, info)：info 含 silhouette/n_clusters/sizes。
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels, dtype=int)
    ks = [k for k in np.unique(labels) if k >= 0]
    fig, ax = new_figure(width, ratio=0.72)
    if (labels == -1).any():
        noise = X[labels == -1]
        ax.plot(noise[:, 0], noise[:, 1], "o", color="0.75", markersize=3,
                alpha=0.6, label=f"噪声（{len(noise)}）")
    sizes = {}
    for idx, k in enumerate(ks):
        pts = X[labels == k]
        sizes[k] = len(pts)
        c = PALETTE[idx % len(PALETTE)]
        name = cluster_names[idx] if cluster_names else f"簇{k}"
        ax.plot(pts[:, 0], pts[:, 1], "o", color=c, markersize=3.6,
                alpha=0.75, markeredgecolor="white", markeredgewidth=0.3,
                label=f"{name}（n={len(pts)}）")
        _cov_ellipse(ax, pts, c, n_std=n_std)
        cx, cy = pts.mean(axis=0)
        ax.plot(cx, cy, "*", color=c, markersize=13,
                markeredgecolor="white", markeredgewidth=0.8, zorder=5)
    info = dict(silhouette=_silhouette(X, labels), n_clusters=len(ks),
                sizes=sizes)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right", fontsize=6.5)
    sil_line = (f"平均轮廓系数 = {info['silhouette']:.2f}"
                if np.isfinite(info["silhouette"])
                else "轮廓系数无定义（簇数 < 2）")
    stat_box(ax, [f"k = {len(ks)} 簇，n = {int((labels >= 0).sum())}",
                  sil_line,
                  f"星标 = 簇心，虚线 = {n_std:g}σ 协方差椭圆"],
             loc="lower right", fontsize=6.5)
    fig._ff_stats = info
    return fig, ax, info


if __name__ == "__main__":
    apply_style()
    rng = np.random.default_rng(17)
    c1 = rng.normal([2.0, 6.0], [0.7, 0.5], (140, 2))
    c2 = rng.normal([6.5, 4.0], [0.9, 0.7], (170, 2))
    c3 = rng.normal([3.5, 1.8], [0.5, 0.6], (110, 2))
    noise = rng.uniform([0, 0], [9, 8], (18, 2))
    X = np.vstack([c1, c2, c3, noise])
    labels = np.r_[np.zeros(140), np.ones(170), np.full(110, 2),
                   np.full(18, -1)].astype(int)
    fig, ax, info = cluster_scatter(
        X, labels, xlabel="人均消费（千元）", ylabel="到访频次（次/月）",
        cluster_names=["高频低消", "低频高消", "低频低消"])
    ax.set_title(f"客群分 {info['n_clusters']} 簇结构清晰"
                 f"（轮廓系数 {info['silhouette']:.2f}），簇间无重叠",
                 fontsize=9.5)
    run_qa(fig, expect_width=("onehalf",))
    save_figure(fig, str(GALLERY / "cluster_scatter"))
    print("cluster_scatter: OK")
