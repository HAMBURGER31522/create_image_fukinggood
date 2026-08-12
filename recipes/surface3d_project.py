"""3D 响应面 + 底面等高投影 + 最优点标注。

论点合同示例：
- 结论：RMS 曲面沿 (D0, ε) 单谷，最优 (0.40, 7e-4)，谷底平坦方向为 ε。
- 证据链：曲面形状 → 底面投影等高（可读等值） → 最优星标 → 参数框。
参考：skills/photo 图 12（双参数响应曲面）。
替代：默认 plot_surface 彩虹面无投影（平庸）。
"""
from _common import GALLERY
import numpy as np
import matplotlib.pyplot as plt

from core import (apply_style, MM, COLUMN_WIDTHS, save_figure, run_qa,
                  cmap_for)


def surface_with_projection(X, Y, Z, best=None, xlabel="x", ylabel="y",
                            zlabel="z", width="onehalf", elev=25, azim=-60,
                            stat_lines=None):
    w = COLUMN_WIDTHS[width] * MM
    fig = plt.figure(figsize=(w, w * 0.85))
    ax = fig.add_subplot(projection="3d")
    fig.subplots_adjust(left=0.0, right=0.95, bottom=0.06, top=0.92)
    ax.plot_surface(X, Y, Z, cmap=cmap_for("surface"), linewidth=0,
                    antialiased=True, alpha=0.9, rstride=2, cstride=2)
    zmin = np.min(Z) - 0.35 * (np.max(Z) - np.min(Z))
    ax.contour(X, Y, Z, levels=12, zdir="z", offset=zmin,
               cmap=cmap_for("surface"), linewidths=0.8)
    if best is not None:
        bx, by = best
        bz = Z[np.unravel_index(np.argmin(Z), Z.shape)]
        ax.plot([bx], [by], [zmin], "*", color="#D55E00", markersize=12,
                markeredgecolor="white", markeredgewidth=0.5, zorder=10)
        ax.plot([bx, bx], [by, by], [zmin, bz], ":", color="#D55E00",
                linewidth=0.9)
    ax.set_zlim(zmin, np.max(Z))
    ax.set_xlabel(xlabel, fontsize=8, labelpad=2)
    ax.set_ylabel(ylabel, fontsize=8, labelpad=2)
    ax.set_zlabel(zlabel, fontsize=8, labelpad=2)
    ax.tick_params(labelsize=6.5, pad=1)
    ax.view_init(elev=elev, azim=azim)
    ax.xaxis.pane.set_alpha(0.05)
    ax.yaxis.pane.set_alpha(0.05)
    ax.zaxis.pane.set_alpha(0.05)
    if stat_lines:
        ax.text2D(0.02, 0.98, "\n".join(stat_lines), transform=ax.transAxes,
                  fontsize=6.5, va="top", linespacing=1.5,
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                            edgecolor="0.7", alpha=0.85, linewidth=0.6))
    return fig, ax


if __name__ == "__main__":
    apply_style()
    d0 = np.linspace(0.34, 0.46, 60)
    eps = np.linspace(4, 10, 60)
    X, Y = np.meshgrid(d0, eps)
    Z = 4.67 + 900 * (X - 0.40) ** 2 + 0.012 * (Y - 7) ** 2
    fig, ax = surface_with_projection(
        X, Y, Z, best=(0.40, 7),
        xlabel="顶点径向偏移 D₀（m）",
        ylabel="间距容差 ε（×10⁻⁴）",
        zlabel="RMS（cm）",
        stat_lines=["网格 60×60，每点内层 QP+SLP",
                    "最优 D₀ = 0.40，ε = 7×10⁻⁴",
                    "RMS = 4.671 cm，谷底沿 ε 平坦"])
    ax.set_title("双参数响应曲面单谷：精度对 D₀ 敏感、对 ε 平坦",
                 fontsize=9, pad=-2)
    save_figure(fig, str(GALLERY / "surface3d_project"))
    run_qa(fig, expect_width=("onehalf",))
    print("surface3d_project: OK")
