"""响应面 + 前沿叠加 与 目标空间 Pareto 前沿。

archetype 1: response_overlay —— 概率/成本响应面(热力) + 多方法前沿线 + 等值线 + 最优星标
archetype 2: pareto_front    —— 目标空间散点 + 非支配前沿连线 + 膝点引线

论点合同示例（pareto）：
- 结论：膝点方案以 +4% 成本换 -31% 误差，是性价比拐点。
- 证据链：支配点灰化 → 前沿阶梯线 → 膝点星标+双目标值引线框。
"""
from _common import GALLERY
import numpy as np

from core import (smart_legend, apply_style, new_figure, save_figure, run_qa,
                  stat_box, callout, cmap_for, semantic, truncate_cmap)


def response_overlay(X, Y, P, fronts, best=None, levels=(0.5, 0.9),
                     xlabel="x", ylabel="y", zlabel="导通概率",
                     width="onehalf"):
    """fronts: [(名称, x, y, 样式dict), ...] 叠加在响应面上的方法对照线。"""
    fig, ax = new_figure(width, ratio=0.75)
    # 截断色图两端高饱和段 + alpha，避免大面积深色压过前沿主线
    pm = ax.pcolormesh(X, Y, P, cmap=truncate_cmap(cmap_for("heatmap")),
                       shading="auto", rasterized=True, alpha=0.85)
    cb = fig.colorbar(pm, ax=ax, pad=0.02, shrink=0.9)
    cb.set_label(zlabel, fontsize=8)
    cb.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cb.ax.tick_params(labelsize=7)
    cs = ax.contour(X, Y, P, levels=list(levels), colors="0.25",
                    linewidths=0.9, linestyles=["--", ":"])
    ax.clabel(cs, inline=True, fontsize=6.5, fmt="%.2f")
    for name, fx, fy, st in fronts:
        ax.plot(fx, fy, label=name, zorder=4, **st)
    if best is not None:
        ax.plot(*best, "*", color="#FFD24C", markeredgecolor="k",
                markersize=13, markeredgewidth=0.8, zorder=6,
                clip_on=False, label="最低成本点")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    smart_legend(ax)
    return fig, ax


def pareto_front(f1, f2, labels=("目标1", "目标2"), knee=None,
                 minimize=(True, True), width="single"):
    """f1/f2 为候选解的两目标值；自动识别非支配集并连阶梯线。"""
    pts = np.c_[f1 if minimize[0] else -np.asarray(f1),
                f2 if minimize[1] else -np.asarray(f2)]
    order = np.argsort(pts[:, 0])
    nd = []
    best2 = np.inf
    for i in order:
        if pts[i, 1] < best2:
            nd.append(i)
            best2 = pts[i, 1]
    nd = np.array(nd)
    dom = np.setdiff1d(np.arange(len(f1)), nd)

    fig, ax = new_figure(width, ratio=0.8)
    ax.plot(np.asarray(f1)[dom], np.asarray(f2)[dom], "o", color="0.78",
            markersize=3.5, label="被支配解", zorder=2)
    fx = np.asarray(f1)[nd]
    fy = np.asarray(f2)[nd]
    ax.step(fx, fy, where="post", color=semantic("data"), linewidth=1.3,
            zorder=3)
    ax.plot(fx, fy, "o", color=semantic("data"), markersize=4.5,
            markeredgecolor="white", markeredgewidth=0.7,
            label="Pareto 前沿", zorder=4)
    if knee is None and len(nd) > 2:
        # 到理想点归一化距离最近者为膝点
        nx = (fx - fx.min()) / (np.ptp(fx) or 1)
        ny = (fy - fy.min()) / (np.ptp(fy) or 1)
        knee = int(np.argmin(np.hypot(nx, ny)))
    if knee is not None:
        ax.plot(fx[knee], fy[knee], "*", color=semantic("highlight"),
                markersize=14, markeredgecolor="white", markeredgewidth=0.6,
                zorder=5)
        callout(ax, xy=(fx[knee], fy[knee]),
                text=f"膝点\n({fx[knee]:.3g}, {fy[knee]:.3g})",
                xytext=(0.62, 0.68), textcoords="axes fraction",
                color=semantic("highlight"), rad=0.25)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    smart_legend(ax)
    stat_box(ax, [f"候选 {len(f1)}，非支配 {len(nd)}"], loc="lower left",
             fontsize=6.5)
    info = dict(fx=fx, fy=fy, knee=knee)
    return fig, ax, info


if __name__ == "__main__":
    apply_style()
    x = np.linspace(0, 0.95, 160)
    y = np.linspace(0, 40, 160)
    X, Y = np.meshgrid(x, y)
    P = 1 / (1 + np.exp(-(X / 0.9 + Y / 36 - 0.62) * 9))
    fx = np.linspace(0, 0.9, 10)
    fy = 36.3 * (1 - (fx / 0.9) ** 0.55)
    fronts = [
        ("剖面法约束前沿", fx, fy,
         dict(color="k", marker="o", markersize=3.5, linewidth=1.3)),
        ("增量法整数前沿", fx[6:], fy[6:] * 0.96,
         dict(color="#2E5E8C", marker="s", markersize=3.5, linewidth=1.2,
              linestyle="--")),
    ]
    fig, ax = response_overlay(
        X, Y, P, fronts, best=(0.87, 0.4),
        xlabel="介质A 体积分数（%）", ylabel="介质B 体积分数（%）")
    ax.set_title("两方法前沿在 A 富端重合，最低成本点贴 P = 0.9 等值线",
                 fontsize=9, pad=8)
    stat_box(ax, [f"响应面网格 {P.shape[0]}×{P.shape[1]}",
                  f"P > 0.9 区域占 {np.mean(P > 0.9):.0%}"],
             loc="lower left", fontsize=6.5)
    run_qa(fig, expect_width=("onehalf",))
    save_figure(fig, str(GALLERY / "front_overlay"))

    rng = np.random.default_rng(4)
    cost = rng.uniform(8, 30, 60)
    err = 40 / (cost - 6) + rng.uniform(0, 2.2, 60)
    fig, ax, info = pareto_front(cost, err,
                                 labels=("总成本（元）", "RMS 误差（cm）"))
    # 硬规则：图题数字来自计算变量（膝点 vs 前沿最低成本端）
    k = info["knee"]
    dc = (info["fx"][k] - info["fx"][0]) / info["fx"][0]
    de = (info["fy"][k] - info["fy"][0]) / info["fy"][0]
    ax.set_title(f"膝点以 +{dc:.0%} 成本换 {de:.0%} 误差", fontsize=9.5)
    run_qa(fig, expect_width=("single",))
    save_figure(fig, str(GALLERY / "pareto_front"))
    print("front_overlay: 2 figures OK")
