"""构图层：边缘分布、小倍数、面板标签、放大子图。"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .style import MM, COLUMN_WIDTHS


def _width_mm(width) -> float:
    return COLUMN_WIDTHS.get(width, width) if isinstance(width, str) else width


def marginal_grid(width="onehalf", ratio: float = 0.85, right: bool = True,
                  top: bool = True, size: float = 0.22):
    """jointplot 式布局：主轴 + 上/右边缘分布轴。

    返回 (fig, ax_main, ax_top, ax_right)，未启用的为 None。
    边缘轴已隐藏冗余脊线与刻度标签。
    """
    w = _width_mm(width) * MM
    fig = plt.figure(figsize=(w, w * ratio))
    nr = 2 if top else 1
    nc = 2 if right else 1
    hr = ([size, 1] if top else [1])
    wr = ([1, size] if right else [1])
    gs = GridSpec(nr, nc, figure=fig, height_ratios=hr, width_ratios=wr,
                  hspace=0.06, wspace=0.06)
    ax_main = fig.add_subplot(gs[nr - 1, 0])
    ax_top = ax_right = None
    if top:
        ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
        ax_top.tick_params(labelbottom=False)
        ax_top.spines[["left", "top", "right"]].set_visible(False)
        ax_top.set_yticks([])
        ax_top.grid(False)
    if right:
        ax_right = fig.add_subplot(gs[nr - 1, nc - 1], sharey=ax_main)
        ax_right.tick_params(labelleft=False)
        ax_right.spines[["bottom", "top", "right"]].set_visible(False)
        ax_right.set_xticks([])
        ax_right.grid(False)
    return fig, ax_main, ax_top, ax_right


def small_multiples(n: int, ncols: int = 4, width="double", ratio: float = 1.0,
                    **kwargs):
    """小倍数网格（迭代快照、多组对比）。返回 (fig, axes 一维列表)。"""
    if n < 1 or ncols < 1:
        raise ValueError(f"n 与 ncols 必须 ≥ 1（n={n}, ncols={ncols}）")
    nrows = (n + ncols - 1) // ncols
    w = _width_mm(width) * MM
    cell = w / ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(w, cell * ratio * nrows),
                             squeeze=False, **kwargs)
    flat = axes.ravel().tolist()
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]


def panel_label(ax, label: str, dx: float = -0.08, dy: float = 1.04,
                fontsize: float = 10):
    """面板标签 (a)(b)(c)，粗体，轴左上角外侧。"""
    ax.text(dx, dy, f"({label})", transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold", va="bottom", ha="right")


def inset_zoom(ax, bounds, xlim, ylim, edgecolor: str = "#C44E52"):
    """放大子图：bounds=(x0,y0,w,h) 轴分数坐标；返回 inset 轴。

    自动画放大区指示框与连接线（连接线弱化，避免喧宾夺主）。
    """
    axins = ax.inset_axes(bounds, xlim=xlim, ylim=ylim)
    axins.set_facecolor("white")
    res = ax.indicate_inset_zoom(
        axins, edgecolor=edgecolor, linewidth=0.9, alpha=0.9)
    # matplotlib <3.10 返回 (rect, connectors)，3.10+ 返回 InsetIndicator
    if isinstance(res, tuple):
        _, connectors = res
    else:
        connectors = res.connectors
    for c in connectors:
        c.set(alpha=0.35, linewidth=0.5, linestyle=":")
    axins.grid(True, linestyle="--", alpha=0.3, linewidth=0.4)
    for s in axins.spines.values():
        s.set_edgecolor(edgecolor)
        s.set_linewidth(0.9)
        s.set_visible(True)
    axins.tick_params(labelsize=6.5)
    return axins
