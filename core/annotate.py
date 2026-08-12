"""注释层：统计注释框、引线标注、线端直标、参考线。

参考图的核心气质来源：图内自含统计证据，读者不用查图例/正文。
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory


def stat_box(ax, lines, loc: str = "upper left", fontsize: float = 7.0,
             edgecolor: str = "0.7", facecolor: str = "white", alpha: float = 0.85,
             pad: float = 0.45):
    """统计注释框。lines 为 str 列表，如 ['n = 692', 'RMS = 5.068 cm']。

    loc: upper/lower + left/right/center，同 legend 习惯。
    """
    text = "\n".join(lines)
    _pos = {
        "upper left": (0.02, 0.98, "left", "top"),
        "upper right": (0.98, 0.98, "right", "top"),
        "lower left": (0.02, 0.02, "left", "bottom"),
        "lower right": (0.98, 0.02, "right", "bottom"),
        "upper center": (0.5, 0.98, "center", "top"),
        "lower center": (0.5, 0.02, "center", "bottom"),
    }[loc]
    x, y, ha, va = _pos
    return ax.text(
        x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=fontsize,
        linespacing=1.5,
        bbox=dict(boxstyle=f"round,pad={pad}", facecolor=facecolor,
                  edgecolor=edgecolor, alpha=alpha, linewidth=0.6),
        zorder=10,
    )


def callout(ax, xy, text, xytext, color: str = "#3D7A6B", fontsize: float = 7.0,
            rad: float = 0.25, textcoords: str = "data"):
    """引线标注：弧形箭头指向关键点（极值、拐点、最优解）。

    xy: 数据坐标目标点；xytext: 文字位置（默认数据坐标，可用 'axes fraction'）。
    """
    return ax.annotate(
        text, xy=xy, xytext=xytext, textcoords=textcoords,
        fontsize=fontsize, color=color, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor=color, alpha=0.9, linewidth=0.7),
        arrowprops=dict(arrowstyle="-|>", color=color, linewidth=0.7,
                        connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=3),
        zorder=11,
    )


def end_label(ax, x, y, text, color, dx: float = 0.008, fontsize: float = 7.5,
              fontweight: str = "bold"):
    """线端直接标注（替代图例）：在曲线末端右侧写名字/数值。"""
    return ax.annotate(
        text, xy=(x, y), xytext=(dx * 72, 0), textcoords="offset points",
        color=color, fontsize=fontsize, fontweight=fontweight,
        ha="left", va="center", zorder=10, annotation_clip=False,
    )


def ref_line(ax, value, orientation: str = "h", label: str | None = None,
             color: str = "0.45", fontsize: float = 7.0,
             label_loc: str = "right"):
    """参考线 + 端点小标签（如 90% 阈值线）。label_loc: left/right。"""
    if orientation == "h":
        ax.axhline(value, color=color, linewidth=0.8, linestyle=(0, (4, 3)), zorder=2)
        if label:
            tf = blended_transform_factory(ax.transAxes, ax.transData)
            xa, ha = (0.995, "right") if label_loc == "right" else (0.01, "left")
            ax.text(xa, value, label, transform=tf, ha=ha, va="bottom",
                    fontsize=fontsize, color=color)
    else:
        ax.axvline(value, color=color, linewidth=0.8, linestyle=(0, (4, 3)), zorder=2)
        if label:
            tf = blended_transform_factory(ax.transData, ax.transAxes)
            ax.text(value, 0.99, label, transform=tf, ha="left", va="top",
                    rotation=90, fontsize=fontsize, color=color)
