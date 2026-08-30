"""构图层：多面板装配器、边缘分布、小倍数、面板标签、放大子图、共享色标。

一张期刊 Figure 不是"一个 chart"，是**一次多面板装配**：主面板大、辅面板小、
共享色标、a/b/c 串成一条论证线、整体读序左→右上→下。figure() 就是这件事的
API；单轴单图只在论点确实只有一层时才用。
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from .style import MM, COLUMN_WIDTHS, MAX_HEIGHT_MM, current_preset, preset_cfg

# Nature: "a full-page figure should probably comprise no more than six panels"
MAX_PANELS = 6


def _width_mm(width) -> float:
    return COLUMN_WIDTHS.get(width, width) if isinstance(width, str) else width


def figure(rows, width="double", height=None, row_heights=None,
           hspace: float = 0.38, wspace: float = 0.30, label: bool = True):
    """多面板装配器：按内容需要分配面板尺寸，而非把所有面板等分。

    Nature 明文要求"面板尺寸反映内容需要、尽量压缩留白"，并按字母序排列。
    等分网格（小倍数除外）是构图偷懒的典型signature。

    rows: 每行一个列表，元素为 名字 或 (名字, 相对宽度)::

        fig, ax = figure([
            [("field", 2), ("hist", 1)],      # 主场图占 2/3 宽
            [("curve", 1), ("resid", 1)],
        ], width="double", row_heights=[1.4, 1])
        ax["field"].contourf(...)

    height: 图高 mm；缺省按行数估算并夹到 Nature 上限 170mm。
    row_heights: 各行相对高度，缺省等高。
    label=True 时按阅读顺序自动打 a/b/c 面板标签。

    返回 (fig, {名字: ax})。名字为 None 的位置留空（用于错位布局）。
    """
    norm = []
    for r in rows:
        row = []
        for item in r:
            name, w = item if isinstance(item, (tuple, list)) else (item, 1)
            if w <= 0:
                raise ValueError(f"面板 '{name}' 相对宽度须 > 0，收到 {w}")
            row.append((name, float(w)))
        if not row:
            raise ValueError("行不能为空")
        norm.append(row)
    if not norm:
        raise ValueError("rows 不能为空")

    names = [n for row in norm for n, _ in row if n is not None]
    if len(names) != len(set(names)):
        dup = {n for n in names if names.count(n) > 1}
        raise ValueError(f"面板名重复：{sorted(dup)}")
    if len(names) > MAX_PANELS:
        print(f"[layout WARN] {len(names)} 个面板超 Nature 建议上限 "
              f"{MAX_PANELS}——拆成两张图，或合并重复信息")

    nrows = len(norm)
    if row_heights is None:
        row_heights = [1.0] * nrows
    if len(row_heights) != nrows:
        raise ValueError(f"row_heights 长度 {len(row_heights)} ≠ 行数 {nrows}")

    w_mm = _width_mm(width)
    h_mm = height if height is not None else min(MAX_HEIGHT_MM,
                                                 w_mm * 0.45 * nrows)
    if h_mm > MAX_HEIGHT_MM:
        print(f"[layout WARN] 图高 {h_mm:.0f}mm 超 Nature 上限 "
              f"{MAX_HEIGHT_MM:.0f}mm，已按上限截断")
        h_mm = MAX_HEIGHT_MM

    fig = plt.figure(figsize=(w_mm * MM, h_mm * MM))
    # 边距按物理尺寸给，而不是用 matplotlib 的默认比例（左.125/右.1/上.12）。
    # 默认值在双栏 183mm 画布上就是十几毫米白边——实测坐标区只占画布
    # 41%–45%，一半以上版面是空的，整图墨迹指标直接被腰斩。
    # 期刊图的边距只需容下轴标签与刻度，约 12–14mm。
    # 右边距要容得下色标**整体**：色标条 + gap 约 9mm 之外，还有它的刻度
    # 标签与轴标题（约 11mm）。只留 9mm 时 share_colorbar(loc="right",
    # label=…) 会把交付宽顶到 190mm、超出 183 档 7mm 而被硬拒——而
    # figure() + share_colorbar 正是 SKILL.md §4b 主推的组合。
    lm, rm = 13.0 / w_mm, 1.0 - 20.0 / w_mm
    bm, tm = 11.0 / h_mm, 1.0 - 6.0 / h_mm
    outer = GridSpec(nrows, 1, figure=fig, height_ratios=row_heights,
                     hspace=hspace, left=lm, right=rm, bottom=bm, top=tm)
    axes: dict = {}
    for i, row in enumerate(norm):
        inner = GridSpecFromSubplotSpec(
            1, len(row), subplot_spec=outer[i],
            width_ratios=[w for _, w in row], wspace=wspace)
        for j, (name, _) in enumerate(row):
            if name is None:
                continue
            axes[name] = fig.add_subplot(inner[0, j])
    if label:
        for k, name in enumerate(names):
            panel_label(axes[name], chr(ord("a") + k))
    return fig, axes


def marginal_grid(width="onehalf", ratio: float = 0.85, right: bool = True,
                  top: bool = True, size: float = 0.22, share: bool = True):
    """jointplot 式布局：主轴 + 上/右边缘分布轴。

    返回 (fig, ax_main, ax_top, ax_right)，未启用的为 None。
    边缘轴已隐藏冗余脊线与刻度标签。

    **share=True（默认）时边缘轴与主轴共享刻度，因此边缘轴上只能放
    主轴同一变量的边缘分布。** 往边缘轴塞别的东西（另一组分类数据、
    另一个量纲）会同时毁掉两个轴：数据被画到视窗外看不见，主轴刻度
    也被改成了不相干的标签。要放别的内容请用 figure() 而不是本函数。
    share=False 解除共享，但那样它就只是个普通两栏布局了。
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
        ax_top = fig.add_subplot(gs[0, 0],
                                 sharex=ax_main if share else None)
        ax_top.tick_params(labelbottom=False)
        ax_top.spines[["left", "top", "right"]].set_visible(False)
        ax_top.set_yticks([])
        ax_top.grid(False)
    if right:
        ax_right = fig.add_subplot(gs[nr - 1, nc - 1],
                                   sharey=ax_main if share else None)
        ax_right.tick_params(labelleft=False)
        ax_right.spines[["bottom", "top", "right"]].set_visible(False)
        ax_right.set_xticks([])
        ax_right.grid(False)
    return fig, ax_main, ax_top, ax_right


def small_multiples(n: int, ncols: int = 4, width="double", ratio: float = 1.0,
                    **kwargs):
    """小倍数网格（迭代快照、多组对比）。返回 (fig, axes 一维列表)。

    小倍数是唯一正当的等分网格：同一编码重复施于不同切片，等分才可比。
    其余多面板场景用 figure() 按内容分配尺寸。
    """
    if n < 1 or ncols < 1:
        raise ValueError(f"n 与 ncols 必须 ≥ 1（n={n}, ncols={ncols}）")
    nrows = (n + ncols - 1) // ncols
    w = _width_mm(width) * MM
    cell = w / ncols
    h_mm = cell * ratio * nrows * 25.4
    if h_mm > MAX_HEIGHT_MM:
        print(f"[layout WARN] 小倍数总高 {h_mm:.0f}mm 超上限 "
              f"{MAX_HEIGHT_MM:.0f}mm，减少行数或压 ratio")
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(w, cell * ratio * nrows),
                             squeeze=False, **kwargs)
    flat = axes.ravel().tolist()
    for ax in flat[n:]:
        ax.set_visible(False)
    # 打标记：小倍数靠"同一编码重复施于不同切片"的可比性取胜，
    # 不靠单面板信息密度。QA 的构成比与图级密度规则对它豁免。
    fig._ff_small_multiples = True
    return fig, flat[:n]


def panel_label(ax, label: str, dx: float = -0.08, dy: float = 1.04,
                fontsize: float | None = None):
    """面板标签。样式随档：

    nature 档 —— Nature 规定"8-pt bold, upright, lowercase a, b, c"，无括号；
    cn 档 —— 中文论文惯例 (a)(b)(c)，10pt 粗体。
    """
    cfg = preset_cfg()
    size = cfg["panel_label_size"] if fontsize is None else fontsize
    text = cfg["panel_label_fmt"].format(
        label.lower() if current_preset() == "nature" else label)
    # Axes3D.text 的签名是 (x, y, z, s)，直接调用会缺参报错；
    # 3D 轴要走 text2D 才能用 transAxes 定位到面板角上
    draw = ax.text2D if hasattr(ax, "text2D") else ax.text
    draw(dx, dy, text, transform=ax.transAxes, fontsize=size,
         fontweight="bold", fontstyle="normal", color="black",
         va="bottom", ha="right")


def share_colorbar(fig, mappable, axes, label: str = "", loc: str = "right",
                   size: float = 0.018, pad: float = 0.015,
                   shrink: float = 1.0):
    """给一组面板挂一条共享色标（多面板必须共享色标才可比）。

    axes: 参与共享的轴列表；色标贴在它们的整体包围盒外侧。
    loc ∈ {right, bottom}。返回 Colorbar。
    """
    axes = list(axes)
    if not axes:
        raise ValueError("axes 不能为空")
    fig.canvas.draw()
    boxes = [a.get_position() for a in axes]
    x0, x1 = min(b.x0 for b in boxes), max(b.x1 for b in boxes)
    y0, y1 = min(b.y0 for b in boxes), max(b.y1 for b in boxes)
    if loc == "right":
        h = (y1 - y0) * shrink
        cax = fig.add_axes([x1 + pad, y0 + (y1 - y0 - h) / 2, size, h])
        cb = fig.colorbar(mappable, cax=cax, orientation="vertical")
    elif loc == "bottom":
        w = (x1 - x0) * shrink
        cax = fig.add_axes([x0 + (x1 - x0 - w) / 2, y0 - pad - size, w, size])
        cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    else:
        raise ValueError(f"loc 只支持 right/bottom，收到 '{loc}'")
    if label:
        cb.set_label(label)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(width=0.4, length=2)
    return cb


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
    if preset_cfg("grid"):
        axins.grid(True, linestyle="-", alpha=0.2, linewidth=0.3)
    else:
        axins.grid(False)
    for s in axins.spines.values():
        s.set_edgecolor(edgecolor)
        s.set_linewidth(0.9)
        s.set_visible(True)
    axins.tick_params(labelsize=max(5.0, plt.rcParams["font.size"] - 2))
    return axins
