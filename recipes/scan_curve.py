"""1D 参数扫描曲线：log y、最优点星标、同色引线框、axvspan 精搜区间。

论点合同示例：
- 结论：目标函数在 D0* = 0.40 m 处唯一谷底，二阶段精搜区间 [0.30, 0.50] 足够。
- 证据链：全局扫描曲线单谷 → 谷底星标+同色统计框 → 精搜带 axvspan。
参考：skills/photo Snipaste_15-27-34（U 形扫描 + 彩色引线框）。
"""
from _common import GALLERY, PRESET
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

from core import (text_color, ptx, smart_legend, apply_style, new_figure, save_figure, run_qa,
                  stat_box, callout, semantic, PALETTE)


def _darken(color, factor=0.62):
    """注释文字用加深色，保证浅色曲线的印刷对比度。"""
    r, g, b = to_rgb(color)
    return (r * factor, g * factor, b * factor)


def scan_curve(x, curves, xlabel, ylabel, logy=True, refine_span=None,
               yticks=None, width="onehalf"):
    """curves: [(名称, y, 颜色), ...]；每条曲线自动星标最小值并加同色注释框。

    yticks: log 轴下显式给刻度（如 [5, 7, 10, 20, 30]），
            避免整个量程只剩一个 10 的幂刻度。
    """
    fig, ax = new_figure(width, ratio=0.55)
    if logy:
        ax.set_yscale("log")
        if yticks is not None:
            ax.yaxis.set_major_locator(FixedLocator(list(yticks)))
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())
    if refine_span:
        ax.axvspan(*refine_span, color="#F3D9DC", alpha=0.5, zorder=1)
        ax.text(np.mean(refine_span), 0.66, "二阶段\n精搜区间",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=ptx(7), color=text_color("#B05661"), linespacing=1.4)
    # 注释框放曲线下方空白区（U 形曲线的左下/中下）
    box_pos = [(0.16, 0.28), (0.48, 0.18)]
    for k, (name, y, c) in enumerate(curves):
        ax.plot(x, y, color=c, linewidth=ptx(1.6, "lw"), label=name, zorder=3)
        i = int(np.argmin(y))
        ax.plot(x[i], y[i], "*", color=c, markersize=ptx(11, "pt"),
                markeredgecolor="white", markeredgewidth=ptx(0.6, "lw"), zorder=4)
        callout(ax, xy=(x[i], y[i]),
                text=f"{name}\nx* = {x[i]:.3g}, min = {y[i]:.4g}",
                xytext=box_pos[k % len(box_pos)],
                textcoords="axes fraction", color=_darken(c), rad=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    smart_legend(ax)
    return fig, ax


if __name__ == "__main__":
    apply_style(PRESET)
    x = np.linspace(-0.6, 0.6, 61)
    y1 = 4.7 + 60 * (x - 0.398) ** 2 + 3 * np.abs(x - 0.398)
    y2 = 5.2 + 55 * (x - 0.390) ** 2 + 2.5 * np.abs(x - 0.390)
    span = (0.30, 0.50)
    fig, ax = scan_curve(
        x, [("斜照工况", y1, "#E8A0A8"), ("天顶工况", y2, "#7FB285")],
        xlabel="顶点径向偏移 D₀（m）", ylabel="口径内节点拟合 RMS（cm）",
        refine_span=span, yticks=(5, 7, 10, 20, 30))
    stat_box(ax, [f"全局扫描 D₀ ∈ [{x[0]:g}, {x[-1]:g}]，步长 {x[1]-x[0]:.2f} m",
                  "两工况均单谷，一维精搜可行"], outside="top")
    ax.set_title(f"目标函数沿 D₀ 单谷：二阶段精搜区间 "
                 f"[{span[0]:.2f}, {span[1]:.2f}] 充分",
                 fontsize=ptx(9.5))
    run_qa(fig, expect_width=("onehalf",))
    save_figure(fig, str(GALLERY / "scan_curve"))
    print("scan_curve: OK")
