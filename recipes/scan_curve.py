"""1D 参数扫描曲线：log y、最优点星标、同色引线框、axvspan 精搜区间。

论点合同示例：
- 结论：目标函数在 D0* = 0.40 m 处唯一谷底，二阶段精搜区间 [0.30, 0.50] 足够。
- 证据链：全局扫描曲线单谷 → 谷底星标+同色统计框 → 精搜带 axvspan。
参考：skills/photo Snipaste_15-27-34（U 形扫描 + 彩色引线框）。
"""
from _common import GALLERY
import numpy as np

from core import (apply_style, new_figure, save_figure, run_qa,
                  stat_box, callout, semantic, PALETTE)


def scan_curve(x, curves, xlabel, ylabel, logy=True, refine_span=None,
               width="onehalf"):
    """curves: [(名称, y, 颜色), ...]；每条曲线自动星标最小值并加同色注释框。"""
    fig, ax = new_figure(width, ratio=0.55)
    if logy:
        ax.set_yscale("log")
    if refine_span:
        ax.axvspan(*refine_span, color="#F3D9DC", alpha=0.5, zorder=1)
        ax.text(np.mean(refine_span), 0.66, "二阶段\n精搜区间",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=7, color="#B05661", linespacing=1.4)
    # 注释框放曲线下方空白区（U 形曲线的左下/中下）
    box_pos = [(0.16, 0.28), (0.48, 0.18)]
    for k, (name, y, c) in enumerate(curves):
        ax.plot(x, y, color=c, linewidth=1.6, label=name, zorder=3)
        i = int(np.argmin(y))
        ax.plot(x[i], y[i], "*", color=c, markersize=11,
                markeredgecolor="white", markeredgewidth=0.6, zorder=4)
        callout(ax, xy=(x[i], y[i]),
                text=f"{name}\nx* = {x[i]:.3g}, min = {y[i]:.4g}",
                xytext=box_pos[k % len(box_pos)],
                textcoords="axes fraction", color=c, rad=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right")
    return fig, ax


if __name__ == "__main__":
    apply_style()
    x = np.linspace(-0.6, 0.6, 61)
    y1 = 4.7 + 60 * (x - 0.398) ** 2 + 3 * np.abs(x - 0.398)
    y2 = 5.2 + 55 * (x - 0.390) ** 2 + 2.5 * np.abs(x - 0.390)
    fig, ax = scan_curve(
        x, [("斜照工况", y1, "#E8A0A8"), ("天顶工况", y2, "#7FB285")],
        xlabel="顶点径向偏移 D₀（m）", ylabel="口径内节点拟合 RMS（cm）",
        refine_span=(0.30, 0.50))
    stat_box(ax, ["全局扫描 D₀ ∈ [−0.6, 0.6]，步长 0.02 m",
                  "两工况均单谷，一维精搜可行"], loc="upper left")
    ax.set_title("目标函数沿 D₀ 单谷：二阶段精搜区间 [0.30, 0.50] 充分",
                 fontsize=9.5)
    save_figure(fig, str(GALLERY / "scan_curve"))
    run_qa(fig, expect_width=("onehalf",))
    print("scan_curve: OK")
